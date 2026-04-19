from contextlib import nullcontext
import os
from pathlib import Path
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
    from svd_baseline_experiment.gpt_new_attn_matrix import GPTAttn
else:
    from .gpt_new_attn_matrix import GPTAttn


def encode(example):
    return tok(example["text"])


def test(dataloader, approx_model, original_model, device, rank):
    original_correct = 0
    approx_correct = 0
    attached = False
    print(f"Num batches in test: {len(dataloader)}")

    with torch.no_grad():
        for x in dataloader:
            last_logit = x["input_ids"][:, -1].to("cpu")
            for k in x:
                x[k] = x[k][:, :-1].to(device, non_blocking=device.type == "cuda")

            actual_y = original_model(**x).logits[:, -1:, :]

            if not attached:
                for i, layer in enumerate(original_model.transformer.h):
                    params = {}
                    for name, param in layer.attn.c_attn.named_parameters():
                        if name == "bias":
                            params["bias"] = param
                        else:
                            params["weights"] = param

                    approx_model.transformer.h[i].attn.c_attn = GPTAttn(params, rank=rank).to(device)
                attached = True

            with torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext():
                pred = approx_model(**x).logits[:, -1:, :]

            actual_logit = actual_y.argmax(-1).to("cpu")
            approx_logit = pred.argmax(-1).to("cpu")

            approx_correct += (approx_logit == last_logit).to(torch.float16).mean()
            original_correct += (actual_logit == last_logit).to(torch.float16).mean()

    print("LAMBADA eval\n-----------------")
    print(f"Original distilgpt2 accuracy: {100 * original_correct / len(dataloader)}")
    print(f"SVD baseline accuracy: {100 * approx_correct / len(dataloader)}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(2, os.cpu_count() or 0)
    rank = 128
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
    approx_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).eval()

    test(test_dataloader, approx_model, original_model, device, rank)
