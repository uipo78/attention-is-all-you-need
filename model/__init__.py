import torch.nn as nn

from ._layers import Encoder, Decoder


class Transformer(nn.Module):
    """docstring for Transformer."""
    # TODO: figure out dimensions

    def __init__(self):
        super().__init__()

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
