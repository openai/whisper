### Whisper CPU 性能优化建议（Windows）

本文面向在 CPU 上运行本仓库 Whisper 的场景，给出按“成本→收益”排序的实用优化项，并提供可直接复制的命令行与代码示例。

---

### 结论速览（优先尝试）

- **解码简化**：`--temperature 0 --beam_size 1 --best_of 1`
- **关闭词级时间戳**：`--word_timestamps False`
- **固定语言**：已知语种时加 `--language zh`（跳过自动检测）
- **设置线程数**：`--threads <物理核数或略低>`（如 8）
- **静音跳过更激进**：`--no_speech_threshold 0.8`（按效果微调）
- **仅处理必要片段**：`--clip_timestamps start,end`

示例（PowerShell）：
```bash
python -m whisper .\audio.wav \
  --device cpu --model small \
  --threads 8 \
  --temperature 0 --beam_size 1 --best_of 1 \
  --word_timestamps False \
  --language zh \
  --no_speech_threshold 0.8
```

---

### 1) 运行参数层（零改代码，见效快）

- **解码策略**：在 CPU 上优先使用贪心解码（降低搜索开销）。
  - 配置：`--temperature 0 --beam_size 1 --best_of 1`

- **关闭词级时间戳**：词级对齐会进行额外的注意力/DTW 计算，CPU 上开销明显。
  - 配置：`--word_timestamps False`
  - 相关代码位置：
```401:411:whisper/transcribe.py
if word_timestamps:
    add_word_timestamps(
        segments=current_segments,
        model=model,
        tokenizer=tokenizer,
        mel=mel_segment,
        num_frames=segment_size,
        prepend_punctuations=prepend_punctuations,
        append_punctuations=append_punctuations,
        last_speech_timestamp=last_speech_timestamp,
    )
```

- **固定语言/跳过检测**：自动语言检测仅用前 30s，但仍有前处理与推理开销。
  - 配置：`--language zh`（或其它目标语言）

- **合理设置线程数**：一般设置为物理核数或略低；避免超线程导致的调度开销。
  - 配置：`--threads <N>`
  - 相关参数定义：
```564:566:whisper/transcribe.py
parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
parser.add_argument("--clip_timestamps", type=str, default="0", help="comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process, where the last end timestamp defaults to the end of the file")
```

- **静音跳过阈值**：提高 `--no_speech_threshold` 可减少对“空窗”的计算时间（存在轻微误杀风险，按效果微调）。

- **只处理必要片段**：对长音频，使用 `--clip_timestamps` 按需裁剪。

- **可选：减少控制台输出**：大量 I/O 也会带来微小开销，`--verbose False` 可略降耗（收益有限）。

---

### 2) 模型选择（速度/质量权衡）

- 尽可能选择更小的模型以提升 CPU 速度：`tiny` < `base` < `small` < `medium`。
- 建议从 `small` 起步，质量可接受且显著快于 `medium`；若对速度极致敏感可尝试 `base/tiny`。

---

### 3) 更快的推理后端：Faster-Whisper（CTranslate2）

在 CPU 上，CTranslate2 的 **INT8/INT8_float32** 推理通常显著快于原生 PyTorch。

- 安装：
```bash
pip install faster-whisper
```

- 最小示例：
```python
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")  # 或 "int8_float32"
segments, info = model.transcribe("audio.wav", language="zh", beam_size=1)
text = "".join([s.text for s in segments])
```

- 建议：
  - 首选 `compute_type="int8"`；若担心精度，尝试 `"int8_float32"`。
  - 结合上文“运行参数层”的建议（固定语言、降低搜索、裁剪片段）。

---

### 4) PyTorch 与环境层优化（可选进阶）

- **设置线程环境变量（Windows）**：
  - PowerShell：
    ```powershell
    $env:OMP_NUM_THREADS = "8"
    $env:MKL_NUM_THREADS = "8"
    ```
  - 与 `--threads` 一致或略低，避免过度并行。

- **在代码中设置线程数**：
```python
import torch
torch.set_num_threads(8)
```

- **试用 PyTorch 2.x `torch.compile`（CPU 也可有收益）**：首轮有编译预热开销，适合长音频/批量处理。
```python
import torch
model = torch.compile(model, backend="inductor", mode="reduce-overhead")
```

- **动态量化 Linear 层**（CPU 常见做法，速度可提升；精度影响通常较小）：
```python
import torch
from torch.ao.quantization import quantize_dynamic

model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

- **依赖更新**：使用较新的 PyTorch（含 MKL/OpenMP 优化）与 NumPy，可获得汇编级优化收益。

---

### 5) 与当前仓库参数的衔接

- 本仓库 CLI 已内置关键开关（`--threads`、`--word_timestamps`、`--language`、`--clip_timestamps` 等），无需改代码即可生效。
- CPU 下已自动禁用 FP16；若未来需要，可考虑扩展 BF16 开关（需 CPU 支持）。

---

### 6) 常见取舍与排错

- 模型越小越快，但识别质量下降；优先在 `small` 与 `base` 之间做权衡。
- 提高 `--no_speech_threshold` 可能误判极低音量语音为静音；按素材微调。
- `torch.compile` 首次调用会更慢（编译），在批量任务中才体现收益。
- 动态量化可能对少数语言/场景略降精度，需 A/B 验证。

---

### 7) 推荐“CPU 友好”启动模板

```bash
python -m whisper .\audio.wav \
  --device cpu --model small \
  --threads 8 \
  --temperature 0 --beam_size 1 --best_of 1 \
  --word_timestamps False \
  --language zh \
  --no_speech_threshold 0.8 \
  --output_dir . --output_format all
```

若希望进一步提速，请优先尝试：更小模型（`base/tiny`）或 Faster-Whisper 的 `int8` 推理。


