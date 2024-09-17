import torch
from torch.nn.functional import normalize

from model.block import FeedForward, MultiHeadAttention
from torch import nn

class Decoder(nn.Module):
    def __init__(self, d_model: int = 512, d_q: int = 512 / 8, d_k: int = 512 / 8, d_v: int = 512 / 8, d_ff=2048, nums_heads: int = 8):
        super().__init__()
        self.multi_head_attention1 = MultiHeadAttention(d_model=d_model, d_q=d_q, d_k=d_k, d_v=d_v, nums_heads=8)
        self.multi_head_attention2 = MultiHeadAttention(d_model=d_model, d_q=d_q, d_k=d_k, d_v=d_v, nums_heads=8)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff)

    def forward(self, x, encoder_x):  # x: torch.Tensor(batch_size, seq_len, d_model)
        q, k, v = x, x, x
        future_mask = torch.tril(torch.ones(x.size(1), x.size(1)))
        h = self.multi_head_attention1(q, k, v, mask=future_mask)  # h: torch.Tensor(batch_size, seq_len, d_model)
        x = normalize(x + h, dim=-1)
        h = self.multi_head_attention2(x, encoder_x, encoder_x)
        x = normalize(x + h, dim=-1)
        h = self.feed_forward(x)
        x = normalize(x + h, dim=-1)
        return x






