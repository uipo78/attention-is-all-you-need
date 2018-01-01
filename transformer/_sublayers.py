import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


class MultiHeadAttention(nn.Module):
    """docstring for MultiHeadAttention."""

    def __init__(self, d_model, h, p, mask):
        super().__init__()

        # See top of page 5
        d_k = d_v = d_model // h

        # Rather than constructing a set of W matrices for Q, K, and V,
        # we instead stack all such matrices, resulting in an order 3 tensor.
        self.W_Q = nn.Parameters(Variable(FloatTensor(h, d_model, d_k)))
        self.W_K = nn.Parameters(Variable(FloatTensor(h, d_model, d_k)))
        self.W_V = nn.Parameters(Variable(FloatTensor(h, d_model, d_v)))
        self.W_O = nn.Parameters(Variable(FloatTensor(h * d_v, d_model)))

        self.attention = _ScaledDotProductAttention(p, mask)

    def forward(self, Q, K, V):
        # Q, K, and V are of dimension (length {Q, K, V}, d_model)
        QW, KW, VW = Q.matmul(self.W_Q), K.matmul(self.W_K), K.matmul(self.W_K)

        # h collection in figure 2
        heads, attns = self.attention(QW, KW, VW)

        # Concat step in figure 2
        batch_size, h, Q_len, d_v = heads.size()
        concat = heads.view(batch_size, Q_len, h * d_v)

        # Linear step in figure 2
        x = concat.bmm(self.W_O)

        return x, attns


class _ScaledDotProductAttention(nn.Module):
    """docstring for _ScaledDotProductAttention."""

    def __init__(self, p, mask):
        super().__init__()

        self.p = p
        self.mask = mask

    def forward(self, Q, K, V):
        # MatMul and Scale steps in figure 2
        x = Q.matmul(K.transpose(-2, -1)) / K.size(-1) ** 0.5

        # Optional masking layer
        if self.mask is not None:
            pass

        # Softmax step in figure 2
        # TODO: Why is there a * here?
        *dims, d_k = x.size()
        # We need to reshape x so that we can apply softmax across the row of
        # each h-slice of each batch
        x = F.softmax(x.view(-1, d_k))
        x = x.view(*dims, d_k)

        # Optional dropout regularization
        if self.p is not None:
            x = F.dropout(x, p=p)

        return x.matmul(V), x


class PositionWiseFFN(nn.Module):
    """docstring for PositionWiseFFN."""

    def __init__(self, d_model, d_inner):
        super().__init__()

        self.zeros = FloatTensor(d_model, d_inner).zeros_()
        self.fc1 = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(d_inner, d_model)

    def forward(self, x):
        # Does this need to be wrapped in Variable
        x = torch.max(input=self.zeros, other=self.fc1(x))
        x = self.fc2(x)

        return x


class LayerNorm(nn.Module):
    """docstring for LayerNorm."""
    # TODO: idk if this is right at all

    def __init__(self, d_hidden, epsilon):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(d_hidden), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(d_hidden), requires_grad=True)
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.epsilon) + self.beta
