import torch
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
                 d_ff,
                 epsilon):
        super().__init__()

        if in_vocab_size == out_vocab_size:
            self._shared_weights = True
            self.token_embedding = nn.Embedding(in_vocab_size, d_model)
            self.pos_embedding = PositionalEncoding(in_vocab_size, d_model)
        else:
            self._shared_weights = False
            self.in_token_embedding = nn.Embedding(in_vocab_size, d_model)
            self.in_pos_embedding = PositionalEncoding(in_vocab_size, d_model)
            self.out_token_embedding = nn.Embedding(out_vocab_size, d_model)
            self.out_pos_embedding = PositionalEncoding(out_vocab_size, d_model)
        self.encoder_stack = nn.ModuleList(
            [EncoderLayer(d_model, h, p, d_ff, epsilon)] * n_layers
        )
        self.decoder_stack = nn.ModuleList(
            [DecoderLayer(d_model, h, p, d_ff, epsilon)] * n_layers
        )
        self.out = nn.Sequential(
            nn.Linear(d_model, out_vocab_size),
            nn.Softmax()
        )

    def forward(self, x_seq, x_pos, y_seq, y_pos):

        # Input and positional embedding
        if self._shared_weights:
            x = self.token_embedding(x_seq) + self.pos_embedding(x_pos)
        else:
            x = self.in_token_embedding(x_seq) + self.in_pos_embedding(x_pos)

        # Encoder
        mask = self._make_padding_mask(x_seq, x_seq)
        for encoder_layer in self.encoder_stack:
            x = encoder_layer(x, mask)

        # Output and positional embedding
        if self._shared_weights:
            y = self.token_embedding(y_seq) + self.pos_embedding(y_pos)
        else:
            y = self.out_token_embedding(y_seq) + self.out_pos_embedding(y_pos)

        # Decoder
        masks = self._make_decoder_masks(x_seq, y_seq)
        for decoder_layer in self.decoder_stack:
            y = decoder_layer(y, x, *masks)

        return self.out(y)

    @staticmethod
    def _make_pad_mask(seq_a, seq_b):
        return seq_a.eq(0).unsqueeze(1).expand(*seq_a.size(), seq_b.size(1))

    @classmethod
    def _make_decoder_masks(cls, x_seq, y_seq):
        pad_mask = cls._make_pad_mask(y_seq, y_seq)
        subseq_mask = cls._make_subsequent_mask(y_seq)
        position_mask = (pad_mask + subseq_mask).gt(0).type_as(x_seq)
        pad_mask = cls._make_padding_mask(x_seq, y_seq)

        return position_mask, pad_mask

    @staticmethod
    def _make_subseq_mask(seq):
        upper_tri_ones = torch.ones(seq.size(1), seq.size(1)).triu_(diagonal=1)
        mask = upper_tri_ones.unsqueeze_(0).repeat(seq.size(0), 1, 1)

        return mask
