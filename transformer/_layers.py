import torch
import torch.nn as nn

from torch.autograd import Variable

from ._sublayers import (
    MultiHeadAttention,
    PositionWiseFFN,
    LayerNorm
)


class PositionalEncoding(nn.Module):
    """docstring for PositionalEncoding."""

    def __init__(self, max_seq_len, d_model):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(10000) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe.unsqueeze(0)

        # Positional encoding is NOT a learned embedding.
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + Variable(self.pe[:, :x.size(1)], requires_grad=False)


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

        self.attns = nn.ModuleList([
            MultiHeadAttention(d_model=d_model, h=h, p=p) for _ in range(2)
        ])
        self.norms = nn.ModuleList([
            LayerNorm(d_hidden=d_model, epsilon=epsilon) for _ in range(3)
        ])
        self.pw_ffn = PositionWiseFFN(d_model, d_ff, dropout)

    def forward(self, x, enc, src_mask, target_mask):
        x = self.norms[0](x + self.attns[0](Q=x, K=x, V=x, mask=target_mask))

        # To understand the inputs for masked_multihead, look at section 3.2.3
        # of the paper.
        x = self.norms[1](x + self.attns[1](Q=x, K=enc, V=enc, mask=src_mask))

        x = self.norms[2](x + self.pw_ffn(x))

        return x
