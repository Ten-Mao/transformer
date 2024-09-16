import torch 
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, nums_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.nums_heads = nums_heads
        for i in range(nums_heads):
            setattr(self, f'W_q_{i}', torch.nn.Linear(d_model, d_q))
            setattr(self, f'W_k_{i}', torch.nn.Linear(d_model, d_k))
            setattr(self, f'W_v_{i}', torch.nn.Linear(d_model, d_v))
        self.linear = torch.nn.Linear(self.nums_heads * self.d_v, d_model)

    def forward(self, q, k, v, mask=None):
        heads = []
        for i in range(self.nums_heads):
            Q = getattr(self, f'W_q_{i}')(q)
            K = getattr(self, f'W_k_{i}')(k)
            V = getattr(self, f'W_v_{i}')(v)
            head = self.scaled_dot_product_attention(Q, K, V, mask=mask)
            heads.append(head)
        head = torch.cat(heads, dim=-1)
        return self.linear(head)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_q = Q.size(-1)
        d_k = K.size(-1)

        assert d_q == d_k

        if mask is not None:
            attention_scores = torch.softmax(Q @ K.transpose(-2, -1) / (d_k ** 0.5) * mask, dim=-1)
        else:
            attention_scores = torch.softmax(Q @ K.transpose(-2, -1) / (d_k ** 0.5), dim=-1)
        return attention_scores @ V


class FeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_model, d_ff)
        self.fc2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x