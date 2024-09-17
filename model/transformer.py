import numpy as np
from torch import nn
import torch
from model.decoder import Decoder
from model.encoder import Encoder

class transformer(nn.Module):
    def __init__(self, d_model: int = 512, d_q: int = 512 // 8, d_k: int = 512 // 8, d_v: int = 512 // 8, d_ff=2048, nums_heads: int = 8, vocab_size: int = 10000, max_len: int = 512, block_nums: int = 6):
        super().__init__()
        self.block_nums = block_nums
        for i in range(block_nums):
            setattr(self, f'encoder_{i}', Encoder(d_model=d_model, d_q=d_q, d_k=d_k, d_v=d_v, d_ff=d_ff, nums_heads=nums_heads))
            setattr(self, f'decoder_{i}', Decoder(d_model=d_model, d_q=d_q, d_k=d_k, d_v=d_v, d_ff=d_ff, nums_heads=nums_heads))
        self.embedding = nn.Embedding(vocab_size, d_model)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))  # [d_model/2]
        pos_embedding = position * div_term  # [max_len, d_model/2]
        # 相当于10000^A = e^(ln(10000^A))

        self.positional_embedding = torch.zeros(max_len, d_model)
        # 计算 sin 和 cos
        self.positional_embedding[:, 0::2] = torch.sin(pos_embedding)  # [max_len, d_model/2] -> [max_len, d_model]，更新偶数维度
        self.positional_embedding[:, 1::2] = torch.cos(pos_embedding)  # [max_len, d_model/2] -> [max_len, d_model]，更新奇数维度

        self.linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, y):
        x = self.embedding(x)
        x += self.positional_embedding[:x.size(-1)]
        y = self.embedding(y)
        y += self.positional_embedding[:y.size(-1)]
        for i in range(self.block_nums):
            x = getattr(self, f'encoder_{i}')(x)
        for i in range(self.block_nums):
            y = getattr(self, f'decoder_{i}')(y, x)
        out = self.linear(y)
        out = torch.softmax(out, dim=-1)
        return out