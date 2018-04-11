import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


if torch.cuda.is_available():
    ByteTensor = torch.cuda.ByteTensor
    FloatTensor = torch.cuda.FloatTensor
else:
    ByteTensor = torch.ByteTensor
    FloatTensor = torch.FloatTensor


class MultiHeadAttention(nn.Module):
    """docstring for MultiHeadAttention."""

    def __init__(self, d_model, h, p):
        super().__init__()

        # See top of page 5
        d_k = d_v = d_model // h

        # Rather than constructing a set of W matrices for Q, K, and V,
        # we instead stack all such matrices, resulting in an order 3 tensor.
        self.W_Q = nn.Parameters(Variable(FloatTensor(h, d_model, d_k)))
        self.W_K = nn.Parameters(Variable(FloatTensor(h, d_model, d_k)))
        self.W_V = nn.Parameters(Variable(FloatTensor(h, d_model, d_v)))
        self.W_O = nn.Parameters(Variable(FloatTensor(h * d_v, d_model)))

        self.scaled_dot_attention = _ScaledDotProductAttention(p)

    def forward(self, Q, K, V, mask):
        # Q, K, and V are of dimension (length {Q, K, V}, d_model)
        QW, KW, VW = Q.matmul(self.W_Q), K.matmul(self.W_K), K.matmul(self.W_K)

        # h collection in figure 2
        heads = self.scaled_dot_attention(QW, KW, VW, mask)

        # Concat step in figure 2
        batch_size, h, Q_len, d_v = heads.size()
        concat = heads.view(batch_size, Q_len, h * d_v)

        # Linear step in figure 2
        x = concat.bmm(self.W_O)

        return x


class _ScaledDotProductAttention(nn.Module):
    """docstring for _ScaledDotProductAttention."""

    def __init__(self, dropout):
        super().__init__()

        self.dropout = dropout

    def forward(self, Q, K, V, mask):
        # MatMul and Scale steps in figure 2
        x = Q.matmul(K.transpose(-2, -1)) / K.size(-1) ** 0.5

        # Optional masking layer
        if self.mask is not None:
            # BELOW IS A BUG https://github.com/pytorch/pytorch/issues/3397
            x.masked_fill_(self.mask.type(ByteTensor), -float("inf"))

        # Softmax step in figure 2
        *dims, d_k = x.size()
        # We need to reshape x so that we can apply softmax across the row of
        # each h-slice of each batch
        x = F.softmax(x.view(-1, d_k))
        x = x.view(*dims, d_k)

        # Optional dropout regularization
        if self.p is not None:
            x = F.dropout(x, p=self.dropout)

        return x.matmul(V)


class PositionWiseFFN(nn.Module):
    """docstring for PositionWiseFFN."""

    def __init__(self, d_model, d_ff):
        super().__init__()

        self.zeros = FloatTensor(d_model, d_ff).zeros_()
        self.fc1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Does this need to be wrapped in Variable
        x = torch.max(input=self.zeros, other=self.fc1(x))
        x = self.fc2(x)

        return x


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
