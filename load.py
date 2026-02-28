from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
import sys
from new_attention_module import Autoencoder, FragmentedKQV
from datasets import load_dataset
from torch.utils.data import DataLoader
import os

def encode(example):
    return tok(example["text"], max_length=512, padding="max_length", truncation=True)

def custom_loss(actual, pred, mse_percent):
    mse_amount = mse_percent * nn.functional.mse_loss(pred, actual)
    cos_amount = (1-mse_percent) * (-1*(torch.clamp_min(-1*nn.functional.cosine_similarity(actual, pred, dim=2).nan_to_num(1).mean() + 1, 1e-6)).log() * 1/4).nan_to_num(0)
    loss = cos_amount+mse_amount
    return loss


def train(dataloader, autoencoder_model, original_model, autoencoder, optimizer, epoch_num, scalar):
    total_loss = 0
    cos_sim = 0
    kl_div_total = 0
    num = 0
    print(f"Num batches: {len(dataloader)}")
    for x in dataloader:
        num+=1
        del x["text"]
        for k in x:
            x[k] = x[k].to("cuda", non_blocking=True)
        with torch.no_grad():
            actual_y = original_model(**x).last_hidden_state.nan_to_num(0)
        
        if num % 511 == 0:
            print(f"Done with {num}")
        if num == 1:
            for i, layer in enumerate(original_model.encoder.layer):
                newModel = {}
                need = False
                for param in layer.attention.self.query.parameters():
                    need = True
                    if (param.dim() == 1):
                        newModel["bias"] = param
                    else:
                        newModel["weights"] = param
                
                if need:
                    if not isinstance(autoencoder_model.encoder.layer[i].attention.self.query, FragmentedKQV):
                        autoencoder_model.encoder.layer[i].attention.self.query = FragmentedKQV(autoencoder, newModel, d=64)
                    else:
                        autoencoder_model.encoder.layer[i].attention.self.query.set_autoencoder(autoencoder)
                    need=False
            
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            generated_ids = autoencoder_model(**x)
            pred = generated_ids.last_hidden_state.nan_to_num(0)
            
            loss = custom_loss(actual_y, pred, 0.8)

        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss+=loss.detach()
        with torch.no_grad():
            cos = torch.nn.functional.cosine_similarity(pred, actual_y, dim=-1).mean()
            print(cos.item())
            cos_sim+=cos.item()

            p = torch.log_softmax(pred, dim=-1)
            q = torch.softmax(actual_y, dim=-1)
            kl_div = torch.nn.functional.kl_div(p, q, reduction="batchmean")
            kl_div_total+=kl_div.item()

    print(f"Epoch #{epoch_num}\n-----------------")
    print(f"MSE Loss: {total_loss/len(dataloader)}")
    print(f"Cos Sim: {cos_sim/len(dataloader)}")
    print(f"KL-Div Loss: {kl_div_total/len(dataloader)}")

if __name__ == "__main__":
    #recons_scal = 0.25
    loss_scal = 0.5
    lrs = [1e-4, 3e-4, 1e-3, 3e-3]
    wd = 0.002
    epochs = 10

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    url = "huawei-noah/TinyBERT_General_4L_312D"


    tok = AutoTokenizer.from_pretrained(url, max_length=512)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(url).to(device).eval()
    changed_model = AutoModel.from_pretrained(url).to(device).eval()

    dataset = dataset.map(encode, batched=True, batch_size=1000).with_format(type="torch")
    data = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
           
    #if os.path.exists("auto_encoder.pth"):
    #   auto_encoder.load_state_dict(torch.load("auto_encoder.pth", weights_only=True))

    scalar = torch.amp.GradScaler()
    for lr in lrs:
        auto_encoder = Autoencoder().to("cuda")
        optimizer = torch.optim.Adam(auto_encoder.parameters(), lr, weight_decay=wd)
        for it in range(epochs):
            train(data, changed_model, model, auto_encoder, optimizer, it+1, scalar)
            torch.save(auto_encoder.state_dict(), f"auto_encoder_{lr}.pth")


            
