import torch.nn as nn

from ._sublayers import (
    MultiHeadAttention,
    PositionWiseFFN,
    LayerNorm
)


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

    def forward(self, x, x_encoder):
        x = self.norm_multihead1(x + self.masked_multihead(x))
        x = self.norm_multihead2(x + self.multhead(x, x_encoder))
        x = self.norm(x + self.pw_ffn(x))

        return x
