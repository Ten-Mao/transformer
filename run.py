from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset.MyDataset import MyDataset
from model.transformer import transformer



if __name__ == "__main__":
    dataset = load_dataset("wmt/wmt14", "de-en")
    trainset, validateset, testset = dataset["train"], dataset["validation"], dataset["test"]

    tokenizer_src = BertTokenizer.from_pretrained('bert-base-german-cased')
    tokenizer_tgt = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size_src = tokenizer_src.vocab_size
    vocab_size_tgt = tokenizer_tgt.vocab_size
    pad_token_src = tokenizer_src.pad_token_id
    pad_token_tgt = tokenizer_tgt.pad_token_id

    trainset = MyDataset(trainset, tokenizer_src, tokenizer_tgt)
    validateset = MyDataset(validateset, tokenizer_src, tokenizer_tgt)
    testset = MyDataset(testset, tokenizer_src, tokenizer_tgt)

    trainLoader = DataLoader(trainset, batch_size=32, shuffle=True)
    validateLoader = DataLoader(validateset, batch_size=32, shuffle=True)
    testLoader = DataLoader(testset, batch_size=32, shuffle=True)

    model = transformer(vocab_size_src=vocab_size_src, vocab_size_tgt=vocab_size_tgt)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=pad_token_tgt)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for src, tgt in trainLoader:

        encoder_input = src
        decoder_input = tgt[:, :-1]
        decoder_output = tgt[:, 1:]

        # src的padding mask
        mask_encoder = (encoder_input != pad_token_src)
        mask_encoder = mask_encoder.unsqueeze(-2).expand(-1, encoder_input.size(-1), -1)

        # tgt的padding mask和future mask
        mask_decoder = (decoder_input != pad_token_tgt)
        mask_decoder = mask_decoder.unsqueeze(-2).expand(-1, decoder_input.size(-1), -1)
        future_mask = torch.tril(torch.ones(decoder_input.size(-1), decoder_input.size(-1))).expand(decoder_input.size(0), -1, -1).bool()
        mask_decoder = mask_decoder & future_mask

        optimizer.zero_grad()

        out = model(encoder_input, decoder_input, mask_encoder, mask_decoder)

        loss = loss_func(out.view(-1, vocab_size_tgt), decoder_output.view(-1))

        loss.backward()

        optimizer.step()

