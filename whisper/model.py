import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight,
            None if self.bias is None else self.bias,
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
            self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight, None if bias is None else bias
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)
        self.scale = (n_state // self.n_head) ** -0.25
        self.is_cross = cross_attention

    def forward(
            self,
            x: Tensor,
            xa: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            keys: Tensor = None,
            values: Tensor = None
    ):
        q = self.query(x)

        # if kv_cache is None or xa is None or self.key not in kv_cache:
        #     # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
        #     # otherwise, perform key/value projections for self- or cross-attention as usual.
        #     k = self.key(x if xa is None else xa)
        #     v = self.value(x if xa is None else xa)
        # else:
        #     # for cross-attention, calculate keys and values once and reuse in subsequent calls.
        #     k = kv_cache[self.key]
        #     v = kv_cache[self.value]

        if keys is not None:
            if self.is_cross:
                if keys.numel() == 0:
                    keys = self.key(x if xa is None else xa)
                    values = self.value(x if xa is None else xa)
                k = keys
                v = values
            else:
                k = self.key(x if xa is None else xa)
                v = self.value(x if xa is None else xa)
                k = torch.cat((keys, k), dim=1)
                v = torch.cat((values, v), dim=1)
                keys = k
                values = v
        else:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk, keys, values

    def qkv_attention(
            self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_ctx = q.size(1)
        q = q * self.scale
        k = k * self.scale
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        if qk.dtype != torch.float:
            qk = qk.float()

        w = F.softmax(qk, dim=-1)
        if w.dtype != q.dtype:
            w = w.to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head, False)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head, cross_attention) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
            self,
            x: Tensor,
            xa: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            self_keys: Tensor = None,
            self_values: Tensor = None,
            cross_keys: Tensor = None,
            cross_values: Tensor = None
    ):
        xn, _, self_keys, self_values = self.attn(self.attn_ln(x), mask=mask, keys=self_keys, values=self_values)
        x = x + xn
        if self.cross_attn:
            xn, _, cross_keys, cross_values = self.cross_attn(self.cross_attn_ln(x), xa, keys=cross_keys,
                                                              values=cross_values)
            x = x + xn
        x = x + self.mlp(self.mlp_ln(x))
        return x, self_keys, self_values, cross_keys, cross_values


class AudioEncoder(nn.Module):
    def __init__(
            self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding)

        for block in self.blocks:
            x = block(x)[0]

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
            self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)
        self.n_layer = n_layer

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, self_keys: List[Tensor] = None, self_values: List[Tensor] = None,
                cross_keys: List[Tensor] = None, cross_values: List[Tensor] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        offset = self_keys[0].shape[1] if len(self_keys[0]) > 0 else 0
        x = (
                self.token_embedding(x)
                + self.positional_embedding[offset: offset + x.shape[-1]]
        )

        next_self_keys = []
        next_self_values = []
        next_cross_keys = []
        next_cross_values = []

        for i in range(self.n_layer):
            x, self_key, self_value, ctx_key, ctx_value = self.blocks[i](
                x, xa, mask=self.mask, self_keys=self_keys[i], self_values=self_values[i], cross_keys=cross_keys[i],
                cross_values=cross_values[i]
            )
            if self_key is not None:
                next_self_keys.append(self_key)
            if self_value is not None:
                next_self_values.append(self_value)
            if ctx_key is not None:
                next_cross_keys.append(ctx_key)
            if ctx_value is not None:
                next_cross_values.append(ctx_value)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight, 0, 1))

        return logits, next_self_keys, next_self_values, next_cross_keys, next_cross_values


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self._encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self._decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        self._traced_decoder = None
        self._traced_decoder_first = None
        self._is_encoder_traced = False
        self._is_first_decoder_traced = False
        self._is_decoder_traced = False
        # use the last half layers for alignment by default; see `set_alignment_heads()` below
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2:] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def encoder(self, x: Tensor):
        if not self._is_encoder_traced:
            self._encoder.eval()
            self._encoder = torch.jit.trace(self._encoder, example_inputs=x)
            self._encoder = torch.jit.freeze(self._encoder)
            self._is_encoder_traced = True
            torch.jit.save(self._encoder, "whisper.pt")
        return self._encoder(x)

    def decoder(self, x: Tensor, xa: Tensor, step: int, self_keys: List[Tensor] = None,
                self_values: List[Tensor] = None, cross_keys: List[Tensor] = None, cross_values: List[Tensor] = None):
        # print("step", step)
        if step == 0:
            if not self._is_first_decoder_traced:
                self._traced_decoder_first = torch.jit.trace(self._decoder,
                                                             (x, xa, self_keys, self_values, cross_keys, cross_values))
                self._traced_decoder_first = torch.jit.freeze(self._traced_decoder_first)
                self._is_first_decoder_traced = True
                torch.jit.save(self._traced_decoder_first, "whisper_decoder_1st.pt")
            return self._decoder(x, xa, self_keys, self_values, cross_keys, cross_values)
            return self._traced_decoder_first(x, xa, self_keys, self_values, cross_keys, cross_values)
        if not self._is_decoder_traced:
            self._traced_decoder = torch.jit.trace(self._decoder,
                                                   (x, xa, self_keys, self_values, cross_keys, cross_values))
            self._traced_decoder = torch.jit.freeze(self._traced_decoder)
            self._is_decoder_traced = True
            torch.jit.save(self._traced_decoder, "whisper_decoder.pt")
        return self._traced_decoder(x, xa, self_keys, self_values, cross_keys, cross_values)
        return self._decoder(x, xa, self_keys, self_values, cross_keys, cross_values)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
            self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[Dict[int, Tensor]] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module: int, _, output: Tensor):
            id_of_module = id(module)
            if id_of_module not in cache or output.shape[1] > self._decoder.positional_embedding.shape[0]:
                cache[id_of_module] = output  # save as-is, for the first token or cross attention
            else:
                cache[id_of_module] = torch.cat([cache[id_of_module], output], dim=1).detach()
            return cache[id_of_module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self._decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
