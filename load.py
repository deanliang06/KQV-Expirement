from transformers import AutoModel, AutoTokenizer
import torch
import sys
from new_attention_module import Autoencoder, FragmentedKQV
from datasets import load_dataset
from torch.utils.data import DataLoader
import os

def encode(example):
    return tok(example["text"], max_length=512, padding="max_length", truncation=True)


def train(dataloader, autoencoder_model, original_model, autoencoder, optimizer, loss_fn, epoch_num):
    total_loss = 0
    num = 0
    autoencoder_model.eval()
    original_model.eval()
    print(f"Num batches: {len(dataloader)}")
    for x in dataloader:
        num+=1
        if num%50==0:
            print(f"Done with {num}")
        del x["text"]
        for k in x:
            x[k] = x[k].to("cuda", non_blocking=True)

        original_output = original_model(**x)
        actual_y = original_output.last_hidden_state
        
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
                        autoencoder_model.encoder.layer[i].attention.self.query = FragmentedKQV(auto_encoder, newModel, d=64)
                    need = False
            
        generated_ids = autoencoder_model(**x)
        pred = generated_ids.last_hidden_state
        
        loss = loss_fn(pred, actual_y)
        total_loss+=loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch #{epoch_num} Loss: {loss/len(dataloader)}")

if __name__ == "__main__":
    #recons_scal = 0.25
    loss_scal = 0.5
    lr = 0.003
    wd = 0.002
    epochs = 10

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    url = "huawei-noah/TinyBERT_General_4L_312D"


    tok = AutoTokenizer.from_pretrained(url, max_length=512)
    model = AutoModel.from_pretrained(url, device_map='auto')
    changed_model = AutoModel.from_pretrained(url, device_map='auto')


    dataset = dataset.map(encode).with_format(type="torch")
    data = DataLoader(dataset, batch_size=12) 
           
    auto_encoder = Autoencoder().to("cuda")
    #if os.path.exists("auto_encoder.pth"):
    #   auto_encoder.load_state_dict(torch.load("auto_encoder.pth", weights_only=True))

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(auto_encoder.parameters(), lr, weight_decay=wd)


    for it in range(epochs):
        train(data, changed_model, model, auto_encoder, optimizer, loss_fn, it+1)
        torch.save(auto_encoder.state_dict(), "auto_encoder.pth")


            
