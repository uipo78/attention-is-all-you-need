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

    def __init__(self, d_model, h, p, d_ff, epsilon):
        super().__init__()

        self.multihead = MultiHeadAttention(d_model=d_model, h=h, p=p)
        self.pw_ffn = PositionWiseFFN(d_model=d_model, d_ff=d_ff)
        self.norm_multihead = LayerNorm(d_hidden=d_model, epsilon=epsilon)
        self.norm = LayerNorm(d_hidden=d_model, epsilon=epsilon)

    def forward(self, x, mask):
        x = self.norm_multihead(x + self.multihead(Q=x, K=x, V=x, mask=mask))
        x = self.norm(x + self.pw_ffn(x))

        return x


class DecoderLayer(nn.Module):
    """docstring for DecoderLayer."""

    def __init__(self, d_model, h, p, d_ff, epsilon):
        super().__init__()

        self.masked_multihead = MultiHeadAttention(d_model=d_model, h=h, p=p)
        self.multihead = MultiHeadAttention(d_model=d_model, h=h, p=p)
        self.pw_ffn = PositionWiseFFN(d_model=d_model, d_ff=d_ff)
        self.norm_masked_multihead = LayerNorm(d_hidden=d_model, epsilon=epsilon)
        self.norm_multihead = LayerNorm(d_hidden=d_model, epsilon=epsilon)
        self.norm = LayerNorm(d_hidden=d_model, epsilon=epsilon)

    def forward(self, x, x_encoded, position_mask, pad_mask):
        x_masked_multihead = self.masked_multihead(Q=x,
                                                   K=x,
                                                   V=x,
                                                   mask=position_mask)
        x = self.norm_masked_multihead1(x + x_masked_multihead)

        # To understand the inputs for masked_multihead, look at section 3.2.3
        # of the paper.
        x_multihead = self.multhead(Q=x,
                                    K=x_encoded,
                                    V=x_encoded,
                                    mask=pad_mask)
        # TODO: is this residual correct?
        x = self.norm_multihead(x + x_encoded + x_multihead)

        x = self.norm(x + self.pw_ffn(x))

        return x
