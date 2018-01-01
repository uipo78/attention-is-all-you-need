import torch
import torch.nn as nn


if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


class PositionalEncoding(nn.Module):
    """docstring for PositionalEncoding."""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding.weight.data = self._init_weights(vocab_size, d_model)

    @staticmethod
    def _init_weights(vocab_size, d_model):
        weights = FloatTensor(vocab_size, d_model).zeros_()
        for pos in range(1, vocab_size):
            for i in range(d_model):
                inner = FloatTensor([pos / (int(10e3)**(2 * i / d_model))])
                if i % 2 == 0:
                    weights[pos, i] = torch.sin(inner)
                else:
                    weights[pos, i] = torch.cos(inner)

        return weights

    def forward(self, x):
        return self.embedding(x)
