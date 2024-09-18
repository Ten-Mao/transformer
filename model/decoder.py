import os
import torch
from torch.nn.functional import normalize

from model.block import FeedForward, MultiHeadAttention
from torch import nn

class Decoder(nn.Module):
    def __init__(self, d_model: int = 512, d_q: int = 512 // 8, d_k: int = 512 // 8, d_v: int = 512 // 8, d_ff=2048, nums_heads: int = 8, dropout=0.1):
        super().__init__()
        self.device = torch.device(os.environ["TORCH_DEVICE"])
        self.multi_head_attention1 = MultiHeadAttention(d_model=d_model, d_q=d_q, d_k=d_k, d_v=d_v, nums_heads=8).to(self.device)
        self.multi_head_attention2 = MultiHeadAttention(d_model=d_model, d_q=d_q, d_k=d_k, d_v=d_v, nums_heads=8).to(self.device)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff).to(self.device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_x, mask_src=None, mask_tgt=None):  # x: torch.Tensor(batch_size, seq_len, d_model)
        q, k, v = x, x, x
        h = self.multi_head_attention1(q, k, v, mask=mask_tgt)  # h: torch.Tensor(batch_size, seq_len, d_model)
        x = normalize(self.dropout(x) + h, dim=-1)
        h = self.multi_head_attention2(x, encoder_x, encoder_x, mask=mask_src)
        x = normalize(self.dropout(x) + h, dim=-1)
        h = self.feed_forward(x)
        x = normalize(self.dropout(x) + h, dim=-1)
        return x






