import os
from datasets import load_dataset
import numpy as np
from sacrebleu import corpus_bleu
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
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

    os.environ["TORCH_DEVICE"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    # os.environ["TORCH_DEVICE"] = "cpu"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device(os.environ["TORCH_DEVICE"])

    model = transformer(vocab_size_src=vocab_size_src, vocab_size_tgt=vocab_size_tgt, block_nums=1).to(device)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=pad_token_tgt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    record_path = "record/transformer.txt"
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for src, tgt in tqdm(trainLoader):
            src = src.to(device)
            tgt = tgt.to(device)

            encoder_input = src
            decoder_input = tgt[:, :-1]
            decoder_output = tgt[:, 1:]

            # src的padding mask
            mask_encoder = (encoder_input != pad_token_src)
            mask_encoder = mask_encoder.unsqueeze(-2).expand(-1, encoder_input.size(-1), -1)

            # tgt的padding mask和future mask
            mask_decoder = (decoder_input != pad_token_tgt)
            mask_decoder = mask_decoder.unsqueeze(-2).expand(-1, decoder_input.size(-1), -1)
            future_mask = torch.tril(torch.ones(decoder_input.size(-1), decoder_input.size(-1))).expand(decoder_input.size(0), -1, -1).bool().to(device)
            mask_decoder = mask_decoder & future_mask

            optimizer.zero_grad()

            out = model(encoder_input, decoder_input, mask_encoder, mask_decoder)

            loss = loss_func(out.reshape(-1, vocab_size_tgt), decoder_output.reshape(-1))

            loss.backward()

            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            bos_token_id = tokenizer_tgt.cls_token_id
            eos_token_id = tokenizer_tgt.sep_token_id
            bleu_batch = np.array([])
            batch_num = np.array([])
            for src, tgt in validateLoader:
                encoder_input = src.to(device)
                mask_encoder = (encoder_input != pad_token_src)
                mask_encoder = mask_encoder.unsqueeze(-2).expand(-1, encoder_input.size(-1), -1)
                decoder_input = torch.full((32, 1), bos_token_id, dtype=torch.long, device=device)
                while decoder_input.size(-1) < 512:
                    mask_decoder = (decoder_input != pad_token_tgt)
                    mask_decoder = mask_decoder.unsqueeze(-2).expand(-1, decoder_input.size(-1), -1)
                    future_mask = torch.tril(torch.ones(decoder_input.size(-1), decoder_input.size(-1))).expand(decoder_input.size(0), -1, -1).bool().to(device)
                    mask_decoder = mask_decoder & future_mask

                    out = model(encoder_input, decoder_input, mask_encoder, mask_decoder)

                    out = torch.softmax(out, dim=-1)

                    next_word = torch.argmax(out[:, -1, :], dim=-1, keepdim=True)

                    decoder_input = torch.cat([decoder_input, next_word], dim=-1)

                    if (next_word == eos_token_id).all():
                        break
                
                # 计算BLEU
                output_text = [tokenizer_tgt.decode(i, skip_special_tokens=True) for i in decoder_input.cpu().numpy()]
                src_text = [tokenizer_src.decode(i, skip_special_tokens=True) for i in encoder_input.cpu().numpy()]
                bleu = corpus_bleu(output_text, src_text)
                bleu_batch = np.append(bleu.score)
                batch_num = np.append(src.size(0))
            
            bleu = np.sum(bleu_batch * batch_num) / np.sum(batch_num)
            print(f"Epoch {epoch} BLEU: {bleu}")
            with open(record_path, "a") as f:
                f.write(f"Epoch {epoch} BLEU: {bleu}\n")
    
    model.eval()
    with torch.no_grad():
        bos_token_id = tokenizer_tgt.cls_token_id
        eos_token_id = tokenizer_tgt.sep_token_id
        bleu_batch = np.array([])
        batch_num = np.array([])
        for src, tgt in testLoader:
            encoder_input = src.to(device)
            mask_encoder = (encoder_input != pad_token_src)
            mask_encoder = mask_encoder.unsqueeze(-2).expand(-1, encoder_input.size(-1), -1)
            decoder_input = torch.full((32, 1), bos_token_id, dtype=torch.long, device=device)
            while decoder_input.size(-1) < 512:
                mask_decoder = (decoder_input != pad_token_tgt)
                mask_decoder = mask_decoder.unsqueeze(-2).expand(-1, decoder_input.size(-1), -1)
                future_mask = torch.tril(torch.ones(decoder_input.size(-1), decoder_input.size(-1))).expand(decoder_input.size(0), -1, -1).bool().to(device)
                mask_decoder = mask_decoder & future_mask

                out = model(encoder_input, decoder_input, mask_encoder, mask_decoder)

                out = torch.softmax(out, dim=-1)

                next_word = torch.argmax(out[:, -1, :], dim=-1, keepdim=True)

                decoder_input = torch.cat([decoder_input, next_word], dim=-1)

                if (next_word == eos_token_id).all():
                    break
            # 计算BLEU
            output_text = [tokenizer_tgt.decode(i, skip_special_tokens=True) for i in decoder_input.cpu().numpy()]
            src_text = [tokenizer_src.decode(i, skip_special_tokens=True) for i in encoder_input.cpu().numpy()]
            bleu = corpus_bleu(output_text, src_text)
            bleu_batch = np.append(bleu.score)
            batch_num = np.append(src.size(0))
        
        bleu = np.sum(bleu_batch * batch_num) / np.sum(batch_num)
        print(f"Test BLEU: {bleu}")
        with open(record_path, "a") as f:
            f.write(f"Test BLEU: {bleu}\n")


