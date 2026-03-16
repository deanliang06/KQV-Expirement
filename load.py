from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
import torch
from torch import nn
import sys
from unet_auto import CNNAutoencoder
from gpt_new_attn_matrix import GPTAttn
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchmetrics.text import Perplexity
import os

def encode(example):
    return tok(example["text"])

def decode(text):
    return tok.decode(text, skip_special_tokens=True)

def custom_loss(actual, pred, mse_percent):
    mse_amount = mse_percent * nn.functional.mse_loss(pred, actual)
    cos_amount = ((1-mse_percent)/4) * (-1 * nn.functional.cosine_similarity(pred, actual, dim=-1).nan_to_num(0).mean() + 1)
    loss = cos_amount+mse_amount
    return loss


def test(dataloader, autoencoder_model, original_model, epoch_num, perplexity):
    total_loss = 0
    cos_sim = 0
    total_perplexity = 0
    num = 0
    with torch.no_grad():
        for x in dataloader:
            num+=1
            for k in x:
              x[k] = x[k].to("cuda")

            with torch.autocast(device_type="cuda", dtype=torch.float32):
              y = original_model(**x).logits[:, :-1, :]
              pred = autoencoder_model(**x).logits[:,:-1, :]
              loss = custom_loss(y, pred, 0.8)

              if not torch.isfinite(loss):
                  continue
            target=x["input_ids"][:,1:]

            pscore = perplexity(preds=pred, target=target)
            if not torch.isfinite(pscore) or not torch:
                print("bruh")
                continue

            total_loss+=loss.detach()
            cos = torch.nn.functional.cosine_similarity(pred, y, dim=-1).mean()
            cos_sim+=cos.item()
            total_perplexity+=(pscore.item()/len(dataloader))

    print(f"Epoch Test #{epoch_num}\n-----------------")
    print(f"Average Perplexity: {total_perplexity}")
    print(f"MSE Loss: {total_loss/len(dataloader)}")
    print(f"Cos Sim: {cos_sim/len(dataloader)}")

def train(dataloader, autoencoder_model, original_model, autoencoder, optimizer, epoch_num, scalar, perplexity=None):
    total_loss = 0
    cos_sim = 0
    num = 0
    total_perlexity = 0
    print(f"Num batches: {len(dataloader)}")
    for x in dataloader:
        num+=1
        for k in x:
          x[k] = x[k].to("cuda", non_blocking=True)
        with torch.no_grad():
            actual_y = original_model(**x).logits[:, :-1, :]
        if num == 1:
            for i, layer in enumerate(original_model.transformer.h):
                params={}
                for name, param in layer.attn.c_attn.named_parameters():
                    if name=="bias":
                        params["bias"]=param
                    else:
                        params["weights"]=param

                if not isinstance(autoencoder_model.transformer.h[i].attn.c_attn, GPTAttn):
                    autoencoder_model.transformer.h[i].attn.c_attn = GPTAttn(auto_encoder, params, d=128).to("cuda")
                else:
                    autoencoder_model.transformer.h[i].attn.c_attn.set_autoencoder(autoencoder)

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            pred = autoencoder_model(**x).logits[:, :-1, :]

        pred = pred.to(torch.float32)
        loss = custom_loss(actual_y, pred, 0.8)
        scalar.scale(loss).backward()
        scalar.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(auto_encoder.parameters(), 1.0)


        if not torch.isfinite(grad_norm): #This is pretty hacky i guess
            for name, p in auto_encoder.named_parameters():
                if p.grad is None:
                    continue
                if (p.grad.nan_to_num(0).mean() == 0):
                    print(f"{name} is problematic")
            optimizer.zero_grad(set_to_none=True)
            scalar.update()
            continue

        scalar.step(optimizer)
        scalar.update()
        optimizer.zero_grad(set_to_none=True)


        loss_add=loss.detach()
        with torch.no_grad():
            pnum = perplexity(preds=pred, target=x["input_ids"][:, 1:])
            if not torch.isfinite(pnum):
                print("Perplexity is not finite")
                continue
            total_perlexity+=(pnum.item()/len(dataloader))
            cos = torch.nn.functional.cosine_similarity(pred, actual_y, dim=-1).mean()
            cos_sim+=cos.item()
            total_loss+=loss_add


    print(f"Epoch Train #{epoch_num}\n-----------------")
    print(f"Perplexity: {total_perlexity}")
    print(f"MSE Loss: {total_loss/len(dataloader)}")
    print(f"Cos Sim: {cos_sim/len(dataloader)}")

if __name__ == "__main__":
    #recons_scal = 0.25
    loss_scal = 0.5
    lrs = [3e-4, 1e-4, 1e-3, 3e-3]
    wd = 0.002
    epochs = 10
    N_EMBED = 768

    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    url = "distilgpt2"


    tok = AutoTokenizer.from_pretrained(url)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained(url).to(device).eval()
    changed_model = AutoModelForCausalLM.from_pretrained(url).to(device).eval()
    for p in changed_model.parameters(): p.requires_grad_(False)

    dataset = train_dataset.map(encode, batched=True, batch_size=1000, remove_columns=["text"]).with_format(type="torch")
    t_dataset = test_dataset.map(encode, batched=True, batch_size=1000, remove_columns=["text"]).with_format(type="torch")
    data = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True, collate_fn=data_collator)
    t_data = DataLoader(t_dataset, batch_size=12, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    #if os.path.exists("auto_encoder.pth"):
    #   auto_encoder.load_state_dict(torch.load("auto_encoder.pth", weights_only=True))

    scalar = torch.amp.GradScaler()
    perplexity = Perplexity(ignore_index=tok.pad_token_id).to(device)

    for lr in lrs:
        auto_encoder = CNNAutoencoder(ndim=N_EMBED, d=128).to(device)
        optimizer = torch.optim.Adam(auto_encoder.parameters(), lr, weight_decay=wd)
        for it in range(epochs):
            train(data, changed_model, model, auto_encoder, optimizer, it+1, scalar, perplexity)
            test(t_data, changed_model, model, it+1, perplexity)
