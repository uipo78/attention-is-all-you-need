import torch.nn as nn

from embed import PositionalEncoding
from ._layers import Encoder, Decoder


class Transformer(nn.Module):
    """docstring for Transformer."""
    # TODO: figure out dimensions

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.io_embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(vocab_size, d_model)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc = nn.Sequential(
            nn.Linear(),
            nn.Softmax()
        )

    def forward(self, x, y):
        x_encoder = self.encoder(x)
        x = self.decoder(y, x_encoder)
        x = self.fc(x)

        return x
