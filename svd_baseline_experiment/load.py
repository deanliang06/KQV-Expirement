import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys

from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.text import Perplexity
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


def custom_loss(actual, pred, kl, T=2.0):
    actual = F.softmax(actual / T, dim=-1)
    pred = F.log_softmax(pred / T, dim=-1)

    kl_loss = kl * F.kl_div(pred, actual, reduction="batchmean") * (T * T)
    hard_targ = pred.argmax(-1)
    cross = (1 - kl) * F.cross_entropy(pred, hard_targ)
    return cross + kl_loss


def attach_gpt_wrappers(target_model, source_model, rank=128, device="cuda"):
    for i, layer in enumerate(source_model.transformer.h):
        params = {}
        for name, param in layer.attn.c_attn.named_parameters():
            if name == "bias":
                params["bias"] = param
            else:
                params["weights"] = param

        target_model.transformer.h[i].attn.c_attn = GPTAttn(params, rank=rank).to(device)


def evaluate(dataloader, approx_model, original_model, perplexity, device):
    total_loss = 0
    total_cos = 0
    total_perplexity = 0
    used_batches = 0

    with torch.no_grad():
        for x in dataloader:
            for k in x:
                x[k] = x[k].to(device)

            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
            with autocast_ctx:
                actual = original_model(**x).logits[:, :-1, :]
                pred = approx_model(**x).logits[:, :-1, :]
                loss = custom_loss(actual, pred, 0.8)

            if not torch.isfinite(loss):
                continue

            pscore = perplexity(preds=pred, target=x["input_ids"][:, 1:])
            if not torch.isfinite(pscore):
                continue

            used_batches += 1
            total_loss += loss.detach()
            total_cos += torch.nn.functional.cosine_similarity(pred, actual, dim=-1).mean().item()
            total_perplexity += pscore.item()

    denom = max(used_batches, 1)
    return {
        "batches": used_batches,
        "perplexity": total_perplexity / denom,
        "custom_loss": (total_loss / denom).item() if torch.is_tensor(total_loss) else float(total_loss),
        "cosine_similarity": total_cos / denom,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=min(2, os.cpu_count() or 0))
    args = parser.parse_args()

    url = "distilgpt2"
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    tok = AutoTokenizer.from_pretrained(url)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, return_tensors="pt")
    original_model = AutoModelForCausalLM.from_pretrained(url).to(device).eval()
    svd_model = AutoModelForCausalLM.from_pretrained(url).to(device).eval()
    for p in svd_model.parameters():
        p.requires_grad_(False)

    dataset = test_dataset.map(encode, batched=True, batch_size=1000, remove_columns=["text"]).with_format(type="torch")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=data_collator,
    )
    perplexity = Perplexity(ignore_index=tok.pad_token_id).to(device)

    attach_gpt_wrappers(svd_model, original_model, rank=args.rank, device=device)
    metrics = evaluate(dataloader, svd_model, original_model, perplexity, device)

    print(f"SVD baseline on WikiText-2 test, rank={args.rank}")
    print("-----------------")
    print(f"Used batches: {metrics['batches']}")
    print(f"Average Perplexity: {metrics['perplexity']}")
    print(f"Custom Loss: {metrics['custom_loss']}")
    print(f"Cos Sim: {metrics['cosine_similarity']}")
