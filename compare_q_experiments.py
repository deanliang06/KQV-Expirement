from pathlib import Path
from contextlib import nullcontext
import importlib.util
import sys

from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.text import Perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling


ROOT = Path(__file__).resolve().parent
EMBED_DIR = ROOT / "unet_embedding_bottleneck"
SVD_DIR = ROOT / "svd_baseline_experiment"

for path in (ROOT, EMBED_DIR, SVD_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


root_unet_auto = load_module("root_unet_auto", ROOT / "unet_auto.py")
root_gpt_attn = load_module("root_gpt_attn", ROOT / "gpt_new_attn_matrix.py")
embed_auto = load_module("embed_auto", EMBED_DIR / "unet_embedding_auto.py")
embed_gpt_attn = load_module("embed_gpt_attn", EMBED_DIR / "gpt_new_attn_matrix.py")
svd_gpt_attn = load_module("svd_gpt_attn", SVD_DIR / "gpt_new_attn_matrix.py")


def encode(example):
    return tok(example["text"])


def custom_loss(actual, pred, kl, T=2.0):
    actual = F.softmax(actual / T, dim=-1)
    pred = F.log_softmax(pred / T, dim=-1)

    kl_loss = kl * F.kl_div(pred, actual, reduction="batchmean") * (T * T)
    hard_targ = pred.argmax(-1)
    cross = (1 - kl) * F.cross_entropy(pred, hard_targ)
    return cross + kl_loss


def attach_root_unet_wrappers(target_model, source_model, autoencoder, d=128, device="cuda"):
    for i, layer in enumerate(source_model.transformer.h):
        params = {}
        for name, param in layer.attn.c_attn.named_parameters():
            if name == "bias":
                params["bias"] = param
            else:
                params["weights"] = param

        if not isinstance(target_model.transformer.h[i].attn.c_attn, root_gpt_attn.GPTAttn):
            target_model.transformer.h[i].attn.c_attn = root_gpt_attn.GPTAttn(autoencoder, params, d=d).to(device)
        else:
            target_model.transformer.h[i].attn.c_attn.set_autoencoder(autoencoder)


def attach_embedding_wrappers(target_model, source_model, autoencoder, device="cuda"):
    for i, layer in enumerate(source_model.transformer.h):
        params = {}
        for name, param in layer.attn.c_attn.named_parameters():
            if name == "bias":
                params["bias"] = param
            else:
                params["weights"] = param

        if not isinstance(target_model.transformer.h[i].attn.c_attn, embed_gpt_attn.GPTAttn):
            target_model.transformer.h[i].attn.c_attn = embed_gpt_attn.GPTAttn(autoencoder, params).to(device)
        else:
            target_model.transformer.h[i].attn.c_attn.set_autoencoder(autoencoder)


def attach_svd_wrappers(target_model, source_model, rank=128, device="cuda"):
    for i, layer in enumerate(source_model.transformer.h):
        params = {}
        for name, param in layer.attn.c_attn.named_parameters():
            if name == "bias":
                params["bias"] = param
            else:
                params["weights"] = param

        target_model.transformer.h[i].attn.c_attn = svd_gpt_attn.GPTAttn(params, rank=rank).to(device)


def evaluate_model(dataloader, candidate_model, original_model, perplexity, device):
    total_loss = 0
    total_cos = 0
    total_perplexity = 0
    used_batches = 0

    with torch.no_grad():
        for x in dataloader:
            for k in x:
                x[k] = x[k].to(device)

            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float32) if device == "cuda" else nullcontext()
            with autocast_ctx:
                actual = original_model(**x).logits[:, :-1, :]
                pred = candidate_model(**x).logits[:, :-1, :]
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
    url = "distilgpt2"
    root_checkpoint = ROOT / "auto_encoder.pth"
    embed_checkpoint = EMBED_DIR / "embedding_auto_model.pth"
    svd_rank = 128

    tok = AutoTokenizer.from_pretrained(url)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, return_tensors="pt")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    dataset = dataset.map(encode, batched=True, batch_size=1000, remove_columns=["text"]).with_format(type="torch")
    dataloader = DataLoader(
        dataset,
        batch_size=12,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=data_collator,
    )

    original_model = AutoModelForCausalLM.from_pretrained(url).to(device).eval()
    perplexity = Perplexity(ignore_index=tok.pad_token_id).to(device)

    if root_checkpoint.exists():
        root_model = AutoModelForCausalLM.from_pretrained(url).to(device).eval()
        for p in root_model.parameters():
            p.requires_grad_(False)

        root_autoencoder = root_unet_auto.CNNAutoencoder(ndim=768, d=128).to(device)
        root_autoencoder.load_state_dict(torch.load(root_checkpoint, map_location=device, weights_only=True))
        attach_root_unet_wrappers(root_model, original_model, root_autoencoder, d=128, device=device)
        root_metrics = evaluate_model(dataloader, root_model, original_model, perplexity, device)
        print("Original U-Net experiment")
        print("-----------------")
        for key, value in root_metrics.items():
            print(f"{key}: {value}")
        print()
    else:
        print(f"Skipping original U-Net experiment, missing checkpoint: {root_checkpoint}")
        print()

    if embed_checkpoint.exists():
        embed_model = AutoModelForCausalLM.from_pretrained(url).to(device).eval()
        for p in embed_model.parameters():
            p.requires_grad_(False)

        embed_autoencoder = embed_auto.UNetEmbeddingAutoencoder(ndim=768, embedding_dim=128).to(device)
        embed_autoencoder.load_state_dict(torch.load(embed_checkpoint, map_location=device, weights_only=True))
        attach_embedding_wrappers(embed_model, original_model, embed_autoencoder, device=device)
        embed_metrics = evaluate_model(dataloader, embed_model, original_model, perplexity, device)
        print("Embedding bottleneck U-Net experiment")
        print("-----------------")
        for key, value in embed_metrics.items():
            print(f"{key}: {value}")
        print()
    else:
        print(f"Skipping embedding bottleneck experiment, missing checkpoint: {embed_checkpoint}")
        print()

    svd_model = AutoModelForCausalLM.from_pretrained(url).to(device).eval()
    for p in svd_model.parameters():
        p.requires_grad_(False)

    attach_svd_wrappers(svd_model, original_model, rank=svd_rank, device=device)
    svd_metrics = evaluate_model(dataloader, svd_model, original_model, perplexity, device)
    print(f"SVD baseline experiment, rank={svd_rank}")
    print("-----------------")
    for key, value in svd_metrics.items():
        print(f"{key}: {value}")
