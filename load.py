from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch import nn
import sys
from new_attention_module import Autoencoder, FragmentedKQV
from unet_module import CNNAutoencoder, CNNBasedFrag

from datasets import load_dataset
from torch.utils.data import DataLoader
from torchmetrics.text import Perplexity
import os

def encode(example):
    return tok(example["text"], max_length=512, padding="max_length", truncation=True)

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
    kl_div_total = 0
    total_perplexity = 0
    num = 0
    with torch.no_grad():
        for x in dataloader:
            num+=1
            del x["text"]
            for k in x:
              x[k] = x[k].to("cuda")
            


            with torch.autocast(device_type="cuda", dtype=torch.float16):
              y = original_model(**x).logits[:, :-1, :]
              pred = autoencoder_model(**x).logits[:,:-1, :]
              loss = custom_loss(y, pred, 0.8)

              if not torch.isfinite(loss):
                  continue
            target=x["input_ids"][:,1:]
            total_loss+=loss.detach()
            cos = torch.nn.functional.cosine_similarity(pred, y, dim=-1).mean()
            cos_sim+=cos.item()

            pscore = perplexity(preds=pred, target=target)
            total_perplexity+=(pscore.item()/len(dataloader))
            
    print(f"Epoch Test #{epoch_num}\n-----------------")
    print(f"Average Perplexity: {total_perplexity}")
    print(f"MSE Loss: {total_loss/len(dataloader)}")
    print(f"Cos Sim: {cos_sim/len(dataloader)}")

def train(dataloader, autoencoder_model, original_model, autoencoder, optimizer, epoch_num, scalar, CNN_based=False, perplexity=None):
    total_loss = 0
    cos_sim = 0
    kl_div_total = 0
    num = 0
    print(f"Num batches: {len(dataloader)}")
    for x in dataloader:
        num+=1
        del x["text"]
        for k in x:
          x[k] = x[k].to("cpu", non_blocking=True)
        with torch.no_grad():
            actual_y = original_model(**x).logits[:, :-1, :]
        if num == 1:
            for i, layer in enumerate(original_model.transformer.h):
                for name, param in layer.attn.c_attn.named_parameters():
                  print(name)
                sys.exit()

            #     newModel = {}
            #     need = False
            #     for param in layer.attention.self.query.parameters():
            #         need = True
            #         if (param.dim() == 1):
            #             newModel["bias"] = param
            #         else:
            #             newModel["weights"] = param

            #     if need:
            #         if not isinstance(autoencoder_model.encoder.layer[i].attention.self.query, (FragmentedKQV, CNNBasedFrag)):
            #             if not CNN_based:
            #                 autoencoder_model.encoder.layer[i].attention.self.query = FragmentedKQV(autoencoder, newModel, d=64)
            #             else:
            #                 autoencoder_model.encoder.layer[i].attention.self.query = CNNBasedFrag(autoencoder, newModel, d=64).to("cuda")
            #         else:
            #             autoencoder_model.encoder.layer[i].attention.self.query.set_autoencoder(autoencoder)
            #         need=False

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = autoencoder_model(**x).logits[:, :-1, :]
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


        total_loss+=loss.detach()
        with torch.no_grad():
            cos = torch.nn.functional.cosine_similarity(pred, actual_y, dim=-1).mean()
            cos_sim+=cos.item()

            p = torch.log_softmax(pred, dim=-1)
            q = torch.softmax(actual_y, dim=-1)
            kl_div = torch.nn.functional.kl_div(p, q, reduction="batchmean")
            kl_div_total+=kl_div.item()

    print(f"Epoch Train #{epoch_num}\n-----------------")
    print(f"MSE Loss: {total_loss/len(dataloader)}")
    print(f"Cos Sim: {cos_sim/len(dataloader)}")
    print(f"KL-Div Loss: {kl_div_total/len(dataloader)}")

if __name__ == "__main__":
    #recons_scal = 0.25
    loss_scal = 0.5
    lrs = [3e-4, 1e-4, 1e-3, 3e-3]
    wd = 0.002
    CNN_based = True
    epochs = 10

    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    url = "distilgpt2"


    tok = AutoTokenizer.from_pretrained(url, max_length=512)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(url).to(device).eval()
    changed_model = AutoModelForCausalLM.from_pretrained(url).to(device).eval()
    for p in changed_model.parameters(): p.requires_grad_(False)

    dataset = train_dataset.map(encode, batched=True, batch_size=1000).with_format(type="torch")
    t_dataset = test_dataset.map(encode, batched=True, batch_size=1000).with_format(type="torch")
    data = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    t_data = DataLoader(t_dataset, batch_size=12, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    #if os.path.exists("auto_encoder.pth"):
    #   auto_encoder.load_state_dict(torch.load("auto_encoder.pth", weights_only=True))

    scalar = torch.amp.GradScaler()
    perplexity = Perplexity(ignore_index=tok.pad_token_id).to(device)
    for lr in lrs:
        auto_encoder = None
        if not CNN_based:
            auto_encoder = Autoencoder().to(device)
        else:
            auto_encoder = CNNAutoencoder().to(device)
        optimizer = torch.optim.Adam(auto_encoder.parameters(), lr, weight_decay=wd)
        for it in range(epochs):
            train(data, changed_model, model, auto_encoder, optimizer, it+1, scalar, CNN_based, perplexity)
            #test(t_data, changed_model, model, it+1, perplexity)
            torch.save(auto_encoder.state_dict(), f"auto_encoder_{lr}.pth")