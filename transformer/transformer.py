import torch.nn as nn

from ._layers import PositionalEncoding, EncoderLayer, DecoderLayer


class Transformer(nn.Module):
    """docstring for Transformer."""
    # TODO: figure out dimensions

    def __init__(self,
                 n_layers,
                 vocab_size,
                 d_model,
                 h,
                 p,
                 mask,
                 d_ff,
                 epsilon):
        super().__init__()

        self.io_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEncoding(vocab_size, d_model)
        self.encoder_stack = nn.ModuleList(
            [EncoderLayer(d_model, h, p, mask, d_ff, epsilon)] * n_layers
        )
        self.decoder_stack = nn.ModuleList(
            [DecoderLayer(d_model, h, p, mask, d_ff, epsilon)] * n_layers
        )
        self.fc_softmax = nn.Sequential(
            nn.Linear(),
            nn.Softmax()
        )

    def forward(self, x_seq, x_pos, y_seq, y_pos):

        # Input and positional embedding
        x = self.io_embedding(x_seq) + self.positional_embedding(x_pos)

        # Encoder
        for encoder_layer in self.encoder_stack:
            x = encoder_layer(x)

        # Output and positional embedding
        y = self.io_embedding(y_seq) + self.positional_embedding(y_pos)

        # Decoder
        for decoder_layer in self.decoder_stack:
            y = decoder_layer(y, x)

        return self.fc_softmax(y)
