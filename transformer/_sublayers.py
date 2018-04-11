import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """docstring for MultiHeadAttention."""

    def __init__(self, d_model, h, dropout):
        super().__init__()

        assert d_model % h == 0

        self.Q_lins = nn.ModuleList()
        self.K_lins = nn.ModuleList()
        self.V_lins = nn.ModuleList()
        for _ in range(h):
            # See top of page 5 for reason for d_model // h
            self.Q_lins.append(nn.Linear(d_model, d_model // h))
            self.K_lins.append(nn.Linear(d_model, d_model // h))
            self.V_lins.append(nn.Linear(d_model, d_model // h))

        self.out = nn.Linear(d_model, d_model)

        self.dropout = dropout

    def forward(self, Q, K, V, mask=None):
        heads = []
        for Q_lin, K_lin, V_lin in zip(self.Q_lins, self.K_lins, self.V_lins):
            heads.append(self._scaled_dot_product_attn(Q_lin(Q),
                                                       K_lin(K),
                                                       V_lin(V),
                                                       mask,
                                                       self.dropout))

        x = torch.cat(heads, dim=1)
        x = self.out(x)

        return x

    @staticmethod
    def _scaled_dot_product_attn(Q, K, V, mask=None, dropout=None):
        # MatMul and Scale steps in figure 2
        x = Q.matmul(K.transpose(-2, -1)) / K.size(-1)**0.5

        # Optional masking layer
        if mask is not None:
            # BELOW IS A BUG https://github.com/pytorch/pytorch/issues/3397
            x.masked_fill_(mask, -float("inf"))

        softmax = F.softmax(x)

        # Optional dropout regularization
        if dropout is not None:
            softmax = F.dropout(softmax, p=dropout)

        return softmax.matmul(V)


class PositionWiseFFN(nn.Module):
    """docstring for PositionWiseFFN."""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.module = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.module(x)


class LayerNorm(nn.Module):
    """docstring for LayerNorm."""

    def __init__(self, d_hidden, epsilon=1e-6):
        super().__init__()

        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(d_hidden))
        self.bias = nn.Parameter(torch.zeros(d_hidden))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.weight * (x - mean) / (std + self.epsilon) + self.bias
