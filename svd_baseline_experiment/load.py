import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys

from datasets import load_dataset
import torch
import torch.nn.functional as F
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


def attach_gpt_wrappers(target_model, source_model, rank=128, device="cuda"):
    for i, layer in enumerate(source_model.transformer.h):
        params = {}
        for name, param in layer.attn.c_attn.named_parameters():
            if name == "bias":
                params["bias"] = param
            else:
                params["weights"] = param

        target_model.transformer.h[i].attn.c_attn = GPTAttn(params, rank=rank).to(device)


def evaluate(dataloader, approx_model, original_model, device):
    original_correct = 0.0
    approx_correct = 0.0
    total_mse = 0.0
    used_batches = 0

    with torch.no_grad():
        for x in dataloader:
            last_token = x["input_ids"][:, -1].to("cpu")
            for k in x:
                x[k] = x[k][:, :-1].to(device, non_blocking=device == "cuda")

            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
            with autocast_ctx:
                actual = original_model(**x).logits[:, -1, :]
                pred = approx_model(**x).logits[:, -1, :]
                mse = F.mse_loss(pred.float(), actual.float())

            if not torch.isfinite(mse):
                continue

            used_batches += 1
            total_mse += mse.item()
            actual_logit = actual.argmax(-1).to("cpu")
            approx_logit = pred.argmax(-1).to("cpu")
            approx_correct += (approx_logit == last_token).to(torch.float32).mean().item()
            original_correct += (actual_logit == last_token).to(torch.float32).mean().item()

    denom = max(used_batches, 1)
    return {
        "batches": used_batches,
        "original_lambada_accuracy": 100.0 * original_correct / denom,
        "svd_lambada_accuracy": 100.0 * approx_correct / denom,
        "mean_squared_error": total_mse / denom,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=min(2, os.cpu_count() or 0))
    args = parser.parse_args()

    url = "distilgpt2"
    test_dataset = load_dataset("cimec/lambada", split="test")

    tok = AutoTokenizer.from_pretrained(url, max_length=512)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, return_tensors="pt")
    original_model = AutoModelForCausalLM.from_pretrained(url).to(device).eval()
    svd_model = AutoModelForCausalLM.from_pretrained(url).to(device).eval()
    for p in svd_model.parameters():
        p.requires_grad_(False)

    dataset = test_dataset.map(encode, batched=True, batch_size=1000, remove_columns=["text", "domain"]).with_format(type="torch")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=data_collator,
    )

    attach_gpt_wrappers(svd_model, original_model, rank=args.rank, device=device)
    metrics = evaluate(dataloader, svd_model, original_model, device)

    print(f"SVD baseline on LAMBADA test, rank={args.rank}")
    print("-----------------")
    print(f"Used batches: {metrics['batches']}")
    print(f"Original distilgpt2 LAMBADA accuracy: {metrics['original_lambada_accuracy']}")
    print(f"SVD baseline LAMBADA accuracy: {metrics['svd_lambada_accuracy']}")
    print(f"Mean squared error: {metrics['mean_squared_error']}")
