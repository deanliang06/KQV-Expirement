from contextlib import nullcontext
from pathlib import Path
import os
import sys

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from unet_original.gpt_new_attn_matrix import GPTAttn
    from unet_original.unet_auto import CNNAutoencoder
else:
    from .gpt_new_attn_matrix import GPTAttn
    from .unet_auto import CNNAutoencoder


CHECKPOINT_PATH = Path(__file__).with_name("auto_encoder.pth")


def encode(example):
    return tok(example["text"])


def autocast_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def test(dataloader, autoencoder_model, original_model, autoencoder, device):
    original_correct = 0
    my_correct = 0
    num = 0
    print(f"Num batches in test: {len(dataloader)}")
    with torch.no_grad():
        for x in dataloader:
            num += 1
            last_logit = x["input_ids"][:, -1].to("cpu")
            for k in x:
                x[k] = x[k][:, :-1]
                x[k] = x[k].to(device, non_blocking=device.type == "cuda")
            with torch.no_grad():
                actual_y = original_model(**x).logits[:, -1:, :]
            if num == 1:
                for i, layer in enumerate(original_model.transformer.h):
                    params = {}
                    for name, param in layer.attn.c_attn.named_parameters():
                        if name == "bias":
                            params["bias"] = param
                        else:
                            params["weights"] = param

                    if not isinstance(autoencoder_model.transformer.h[i].attn.c_attn, GPTAttn):
                        autoencoder_model.transformer.h[i].attn.c_attn = GPTAttn(autoencoder, params, d=128).to(device)
                    else:
                        autoencoder_model.transformer.h[i].attn.c_attn.set_autoencoder(autoencoder)

            with autocast_context(device):
                pred = autoencoder_model(**x).logits[:, -1:, :]

            actual_logit = actual_y.argmax(-1).to("cpu")
            my_logit = pred.argmax(-1).to("cpu")

            my_correct += (my_logit == last_logit).to(torch.float16).mean()
            original_correct += (actual_logit == last_logit).to(torch.float16).mean()

    print("LAMBADA eval\n-----------------")
    print(f"Original distilbert accurcy: {100 * original_correct / len(dataloader)}")
    print(f"Our represnetation model accurcy: {100 * my_correct / len(dataloader)}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(2, os.cpu_count() or 0)
    test_dataset = load_dataset("cimec/lambada", split="test")

    tok = AutoTokenizer.from_pretrained("distilgpt2", max_length=512)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, return_tensors="pt")
    test_dataset = test_dataset.map(encode, batched=True, batch_size=1000, remove_columns=["text", "domain"]).with_format(type="torch")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=24,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=data_collator,
    )

    original_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).eval()
    autoencoder_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).eval()
    model = CNNAutoencoder(768, 128).to(device)
    if CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    else:
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")

    test(test_dataloader, autoencoder_model, original_model, model, device)
