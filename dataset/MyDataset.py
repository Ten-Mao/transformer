from torch.utils.data import Dataset
from torch.nn.functional import pad

class MyDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, seq_len=512):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = seq_len
    
    def __getitem__(self, index):
        src = self.dataset[index]["translation"]["de"]
        tgt = self.dataset[index]["translation"]["en"]
        src = self.tokenizer_src(src, return_tensors="pt")["input_ids"].squeeze(0)
        tgt = self.tokenizer_tgt(tgt, return_tensors="pt")["input_ids"].squeeze(0)
        src = pad(src, (0, self.seq_len - src.size(0)), value=self.tokenizer_src.pad_token_id)
        tgt = pad(tgt, (0, self.seq_len - tgt.size(0)), value=self.tokenizer_tgt.pad_token_id)
        return src, tgt

    def __len__(self):
        return len(self.dataset)
