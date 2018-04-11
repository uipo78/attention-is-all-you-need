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

    def __init__(self, d_model, h, p, d_ff, epsilon, dropout):
        super().__init__()

        self.multihead = MultiHeadAttention(d_model=d_model, h=h, p=p)
        self.norms = nn.ModuleList([
            LayerNorm(d_hidden=d_model, epsilon=epsilon) for _ in range(3)
        ])
        self.pw_ffn = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(self, x, mask):
        x = self.norms[0](x + self.multihead(Q=x, K=x, V=x, mask=mask))
        x = self.norms[1](x + self.pw_ffn(x))

        return x


class DecoderLayer(nn.Module):
    """docstring for DecoderLayer."""

    def __init__(self, d_model, h, p, d_ff, epsilon, dropout):
        super().__init__()

        self.multiheads = nn.ModuleList([
            MultiHeadAttention(d_model=d_model, h=h, p=p) for _ in range(2)
        ])
        self.norms = nn.ModuleList([
            LayerNorm(d_hidden=d_model, epsilon=epsilon) for _ in range(3)
        ])
        self.pw_ffn = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(self, x, enc, pos_mask, pad_mask):
        x = self.norms[0](x + self.multiheads[0](Q=x, K=x, V=x, mask=pos_mask))

        # To understand the inputs for masked_multihead, look at section 3.2.3
        # of the paper.
        # TODO: is this residual correct?
        x = self.norms[1](x + self.multiheads[1](Q=x, K=enc, V=enc, mask=pad_mask))

        x = self.norms[2](x + self.pw_ffn(x))

        return x
