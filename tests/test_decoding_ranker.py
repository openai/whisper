import torch

from whisper.decoding import MaximumLikelihoodRanker


def test_maximum_likelihood_ranker_accepts_empty_sequence():
    ranker = MaximumLikelihoodRanker(length_penalty=None)
    tokens = [[torch.empty(0, dtype=torch.long), torch.tensor([1])]]
    sum_logprobs = [[-0.1, -0.2]]

    assert ranker.rank(tokens, sum_logprobs) == [0]
