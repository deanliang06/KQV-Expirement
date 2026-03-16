import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from unet_auto import CNNAutoencoder
import os
import sys


def encode(example):
    return tok(example["text"])

def test(dataloader, autoencoder_model, original_model, autoencoder, scalar):
    original_correct = 0
    my_correct = 0
    num = 0
    print(f"Num batches in test: {len(dataloader)}")
    with torch.no_grad():
        for x in dataloader:
            num+=1
            last_logit = x["input_ids"][:, -1]
            for k in x:
                if k == "input_ids":
                    x[k] = x[k][:-1]
                x[k] = x[k].to("cuda", non_blocking=True)
            with torch.no_grad():
                actual_y = original_model(**x).logits[:, -1]
            if num == 1:
                for i, layer in enumerate(original_model.transformer.h):
                    params={}
                    for name, param in layer.attn.c_attn.named_parameters():
                        if name=="bias":
                            params["bias"]=param
                        else:
                            params["weights"]=param

                    if not isinstance(autoencoder_model.transformer.h[i].attn.c_attn, GPTAttn):
                        autoencoder_model.transformer.h[i].attn.c_attn = GPTAttn(autoencoder, params, d=128).to("cuda")
                    else:
                        autoencoder_model.transformer.h[i].attn.c_attn.set_autoencoder(autoencoder)

            with torch.autocast(device_type="cuda", dtype=torch.float32):
                pred = autoencoder_model(**x).logits[:, -1]
                
            actual_logit = actual_y.argmax(-1)
            my_logit = pred.argmax(-1)

            my_correct += (my_logit == last_logit).mean()
            original_correct += (actual_logit == last_logit).mean()


    print(f"LAMBADA eval\n-----------------")
    print(f"Original distilbert accurcy: {original_correct/len(dataloader)}")
    print(f"Our represnetation model accurcy: {my_correct/len(dataloader)}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = load_dataset("cimec/lambada", split="test")

    tok = AutoTokenizer.from_pretrained("distilgpt2", max_length=512)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, return_tensors="pt")
    test_dataset = test_dataset.map(encode, batched=True, batch_size=1000, remove_columns=["text", "domain"]).with_format(type="torch")
    test_dataloader = DataLoader(test_dataset, batch_size=24, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True, collate_fn=data_collator)

    original_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).eval()
    autoencoder_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).eval()
    model = CNNAutoencoder().to(device) 
    if os.path.exists("auto_model.pth"):
        model.load_state_dict(torch.load("auto_model.pth", weights_only=True))
    else:
        print("BOSS you don't have a model saved")

    scalar = torch.amp.GradScaler()

    test(test_dataloader, autoencoder_model, original_model, model, scalar)


