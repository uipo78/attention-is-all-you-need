import torch.nn as nn

from embed import PositionalEncoding
from ._layers import EncoderLayer, DecoderLayer


class Transformer(nn.Module):
    """docstring for Transformer."""
    # TODO: figure out dimensions

    def __init__(self, vocab_size, d_model, n_layers=6):
        super().__init__()
        self.io_embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(vocab_size, d_model)
        self.encoder_stack = nn.ModuleList([EncoderLayer()] * n_layers)
        self.decoder_stack = nn.ModuleList([DecoderLayer()] * n_layers)
        self.fc = nn.Sequential(
            nn.Linear(),
            nn.Softmax()
        )

    def forward(self, x, y):
        # Encoder
        for layer in self.encoder_stack:
            x = layer(x)

        # Decoder
        for layer in self.decoder_stack:
            x = layer(y, x_encoder)

        x = self.fc(x)

        return x
