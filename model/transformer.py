import os
import numpy as np
from torch import nn
import torch
from model.decoder import Decoder
from model.encoder import Encoder

class transformer(nn.Module):
    def __init__(self, d_model: int = 512, d_q: int = 512 // 8, d_k: int = 512 // 8, d_v: int = 512 // 8, d_ff=2048, nums_heads: int = 8, vocab_size_src: int = 10000, vocab_size_tgt: int = 10000, max_len: int = 512, block_nums: int = 6):
        super().__init__()
        self.device = torch.device(os.environ["TORCH_DEVICE"])
        self.block_nums = block_nums
        for i in range(block_nums):
            setattr(self, f'encoder_{i}', Encoder(d_model=d_model, d_q=d_q, d_k=d_k, d_v=d_v, d_ff=d_ff, nums_heads=nums_heads).to(self.device))
            setattr(self, f'decoder_{i}', Decoder(d_model=d_model, d_q=d_q, d_k=d_k, d_v=d_v, d_ff=d_ff, nums_heads=nums_heads).to(self.device))
        self.src_embedding = nn.Embedding(vocab_size_src, d_model).to(self.device)
        self.tgt_embedding = nn.Embedding(vocab_size_tgt, d_model).to(self.device)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))  # [d_model/2]
        pos_embedding = position * div_term  # [max_len, d_model/2]
        # 相当于10000^A = e^(ln(10000^A))

        self.positional_embedding = torch.zeros(max_len, d_model)
        # 计算 sin 和 cos
        self.positional_embedding[:, 0::2] = torch.sin(pos_embedding)  # [max_len, d_model/2] -> [max_len, d_model]，更新偶数维度
        self.positional_embedding[:, 1::2] = torch.cos(pos_embedding)  # [max_len, d_model/2] -> [max_len, d_model]，更新奇数维度

        self.positional_embedding = self.positional_embedding.to(self.device)

        self.linear = nn.Linear(d_model, vocab_size_tgt).to(self.device)
    
    def forward(self, x, y, mask_encoder=None, mask_decoder=None):
        x = self.src_embedding(x)
        x += self.positional_embedding[:x.size(-2)]
        y = self.tgt_embedding(y)
        y += self.positional_embedding[:y.size(-2)]
        for i in range(self.block_nums):
            x = getattr(self, f'encoder_{i}')(x, mask=mask_encoder)
        for i in range(self.block_nums):
            y = getattr(self, f'decoder_{i}')(y, x, mask_src=mask_encoder[:, :mask_decoder.size(-2), :], mask_tgt=mask_decoder)
        out = self.linear(y)
        return out