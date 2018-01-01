import torch
import torch.nn as nn

from ._sublayers import (
    MultiHeadAttention,
    PositionWiseFFN,
    LayerNorm
)


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


class EncoderLayer(nn.Module):
    """docstring for EncoderLayer."""
    # TODO: figure out dimensions

    def __init__(self, d_model):
        super().__init__()
        self.multihead = MultiHeadAttention()
        self.pw_ffn = PositionWiseFFN()
        self.norm_multihead = LayerNorm(dim_hidden=d_model)
        self.norm = LayerNorm(dim_hidden=d_model)

    def forward(self, x):
        x_multihead, _ = self.multihead(x)
        x = self.norm_multihead(x + x_multihead)
        x = self.norm(x + self.pw_ffn(x))

        return x


class DecoderLayer(nn.Module):
    """docstring for DecoderLayer."""
    # TODO: figure out dimensions

    def __init__(self, d_model):
        super().__init__()
        self.masked_multihead = MultiHeadAttention()
        self.multihead = MultiHeadAttention()
        self.pw_ffn = PositionWiseFFN()
        self.norm_multihead1 = LayerNorm(dim_hidden=d_model)
        self.norm_multihead2 = LayerNorm(dim_hidden=d_model)
        self.norm = LayerNorm(dim_hidden=d_model)

    def forward(self, x, x_encoded):
        x = self.norm_multihead1(x + self.masked_multihead(x))
        x = self.norm_multihead2(x + self.multhead(x, x_encoded))
        x = self.norm(x + self.pw_ffn(x))

        return x
