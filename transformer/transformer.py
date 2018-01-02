import torch.nn as nn

from ._layers import PositionalEncoding, EncoderLayer, DecoderLayer


class Transformer(nn.Module):
    """docstring for Transformer."""

    def __init__(self,
                 n_layers,
                 in_vocab_size,
                 out_vocab_size,
                 d_model,
                 h,
                 p,
                 mask,
                 d_ff,
                 epsilon):
        super().__init__()

        if in_vocab_size == out_vocab_size:
            self._shared_weights = True
            self.io_embedding = nn.Embedding(input_vocab_size, d_model)
            self.pos_embedding = PositionalEncoding(input_vocab_size, d_model)
        else:
            self._shared_weights = False
            self.in_embedding = nn.Embedding(in_vocab_size, d_model)
            self.in_pos_embedding = PositionalEncoding(in_vocab_size, d_model)
            self.out_embedding = nn.Embedding(out_vocab_size, d_model)
            self.out_pos_embedding = PositionalEncoding(out_vocab_size, d_model)
        self.encoder_stack = nn.ModuleList(
            [EncoderLayer(d_model, h, p, mask, d_ff, epsilon)] * n_layers
        )
        self.decoder_stack = nn.ModuleList(
            [DecoderLayer(d_model, h, p, mask, d_ff, epsilon)] * n_layers
        )
        self.fc_softmax = nn.Sequential(
            nn.Linear(d_model, out_vocab_size),
            nn.Softmax()
        )

    def forward(self, x_seq, x_pos, y_seq, y_pos):

        # Input and positional embedding
        if self._shared_weights:
            x = self.io_embedding(x_seq) + self.pos_embedding(x_pos)
        else:
            x = self.in_embedding(x_seq) + self.in_pos_embedding(x_pos)

        # Encoder
        for encoder_layer in self.encoder_stack:
            x = encoder_layer(x)

        # Output and positional embedding
        if self._shared_weights:
            y = self.io_embedding(y_seq) + self.pos_embedding(y_pos)
        else:
            y = self.out_embedding(y_seq) + self.out_pos_embedding(y_pos)

        # Decoder
        for decoder_layer in self.decoder_stack:
            y = decoder_layer(y, x)

        return self.fc_softmax(y)
