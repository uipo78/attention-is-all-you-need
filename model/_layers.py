import torch.nn as nn

from ._sublayers import (
    PositionalEncoding,
    MultiHeadAttention,
    PositionWiseFFN,
    LayerNorm
)

class Encoder(nn.Module):
    """docstring for Encoder."""
    # TODO: figure out dimensions

    def __init__(self, n_layers=6):
        super().__init__()
        self.embedding = nn.Embedding()
        self.pe = PositionalEncoding()
        self.stack = nn.ModuleList([_EncoderLayer()] * n_layers)

    def forward(self, x):
        x = self.pe(x)
        for layer in self.stack:
            pass

        return x


class _EncoderLayer(nn.Module):
    """docstring for _EncoderLayer."""
    # TODO: figure out dimensions

    def __init__(self):
        super().__init__()
        self.multihead = MultiHeadAttention()
        self.pw_ffn = PositionWiseFFN()
        self.norm = LayerNorm()

    def forward(self, x):
        x = self.norm(x + self.multihead(x))
        x = self.norm(x + self.pw_ffn(x))

        return x


class Decoder(nn.Module):
    """docstring for Decoder."""
    # TODO: figure out dimensions

    def __init__(self, n_layers=6):
        super().__init__()
        self.embedding = nn.Embedding()
        self.pe = PositionalEncoding()
        self.stack = nn.ModuleList([_EncoderLayer()] * n_layers)

    def forward(self, x):
        x = self.pe(x)
        for layer in self.stack:
            pass

        return x


class _DecoderLayer(nn.Module):
    """docstring for _DecoderLayer."""
    # TODO: figure out dimensions

    def __init__(self):
        super().__init__()
        self.masked_multihead = MultiHeadAttention()
        self.multihead = MultiHeadAttention()
        self.pw_ffn = PositionWiseFFN()
        self.norm = LayerNorm()

    def forward(self, x, x_encoder):
        x = self.norm(x + self.masked_multihead(x))
        x = self.norm(x + self.multhead(x, x_encoder))
        x = self.norm(x + self.pw_ffn(x))

        return x
