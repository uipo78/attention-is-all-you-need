import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


class PositionalEncoding(nn.Module):
    """docstring for PositionalEncoding."""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.embedding.weight.data = self._init_weights(vocab_size, d_model)

    @staticmethod
    def _init_weights(vocab_size, d_model):
        weights = FloatTensor(vocab_size, d_model).zeros_()
        for pos in range(1, vocab_size):
            for i in range(d_model):
                inner = FloatTensor([pos / (int(10e3)**(2 * i / d_model))])
                if i % 2 == 0:
                    weights[pos, i] = torch.sin(inner)
                else:
                    weights[pos, i] = torch.cos(inner)

        return weights

    def forward(self):
        pass


class MultiHeadAttention(nn.Module):
    """docstring for MultiHeadAttention."""

    def __init__(self, d_model=512, h=8, p=None, mask=None):
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
        heads = self.attention(QW, KW, VW)

        # Concat step in figure 2
        batch_size, h, Q_len, d_v = heads.size()
        concat = heads.view(batch_size, Q_len, h * d_v)

        # Linear step in figure 2
        x = concat.bmm(self.W_O)

        return x


class _ScaledDotProductAttention(nn.Module):
    """docstring for _ScaledDotProductAttention."""

    def __init__(self, p=None, mask=None):
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
        *dims, d_k = x.size()
        # We need to reshape x so that we can apply softmax across the row of
        # each h-slice of each batch
        x = F.softmax(x.view(-1, d_k))
        x = x.view(*dims, d_k)

        # Optional dropout regularization
        if self.p is not None:
            x = F.dropout(x, p=p)

        return x.matmul(V)


class PositionWiseFFN(nn.Module):
    """docstring for PositionWiseFFN."""
    # TODO: figure out dimensions

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(),
            nn.ReLU()
        )
        self.fc2 = nn.Linear()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class LayerNorm(nn.Module):
    """docstring for LayerNorm."""

    def __init__(self, dim_hidden, epsilon=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim_hidden))
        self.beta = nn.Parameter(torch.zeros(dim_hidden))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.epsilon) + self.beta
