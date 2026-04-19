from pathlib import Path
from contextlib import nullcontext
import importlib.util
import sys

from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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


root_unet_auto = load_module("root_unet_auto", ROOT / "unet_original" / "unet_auto.py")
root_gpt_attn = load_module("root_gpt_attn", ROOT / "unet_original" / "gpt_new_attn_matrix.py")
embed_auto = load_module("embed_auto", EMBED_DIR / "unet_embedding_auto.py")
embed_gpt_attn = load_module("embed_gpt_attn", EMBED_DIR / "gpt_new_attn_matrix.py")
svd_gpt_attn = load_module("svd_gpt_attn", SVD_DIR / "gpt_new_attn_matrix.py")


def encode(example):
    return tok(example["text"])


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


def evaluate_model(dataloader, candidate_model, original_model, device):
    original_correct = 0.0
    candidate_correct = 0.0
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
                pred = candidate_model(**x).logits[:, -1, :]
                mse = F.mse_loss(pred.float(), actual.float())

            if not torch.isfinite(mse):
                continue

            used_batches += 1
            total_mse += mse.item()
            actual_logit = actual.argmax(-1).to("cpu")
            candidate_logit = pred.argmax(-1).to("cpu")
            original_correct += (actual_logit == last_token).to(torch.float32).mean().item()
            candidate_correct += (candidate_logit == last_token).to(torch.float32).mean().item()

    denom = max(used_batches, 1)
    return {
        "batches": used_batches,
        "original_lambada_accuracy": 100.0 * original_correct / denom,
        "candidate_lambada_accuracy": 100.0 * candidate_correct / denom,
        "mean_squared_error": total_mse / denom,
    }


if __name__ == "__main__":
    url = "distilgpt2"
    root_checkpoint = ROOT / "unet_original" / "auto_encoder.pth"
    embed_checkpoint = EMBED_DIR / "embedding_auto_model.pth"
    svd_rank = 128

    tok = AutoTokenizer.from_pretrained(url, max_length=512)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, return_tensors="pt")
    dataset = load_dataset("cimec/lambada", split="test")
    dataset = dataset.map(encode, batched=True, batch_size=1000, remove_columns=["text", "domain"]).with_format(type="torch")
    dataloader = DataLoader(
        dataset,
        batch_size=24,
        shuffle=True,
        num_workers=2,
        pin_memory=device == "cuda",
        persistent_workers=True,
        collate_fn=data_collator,
    )

    original_model = AutoModelForCausalLM.from_pretrained(url).to(device).eval()

    if root_checkpoint.exists():
        root_model = AutoModelForCausalLM.from_pretrained(url).to(device).eval()
        for p in root_model.parameters():
            p.requires_grad_(False)

        root_autoencoder = root_unet_auto.CNNAutoencoder(ndim=768, d=128).to(device)
        root_autoencoder.load_state_dict(torch.load(root_checkpoint, map_location=device, weights_only=True))
        attach_root_unet_wrappers(root_model, original_model, root_autoencoder, d=128, device=device)
        root_metrics = evaluate_model(dataloader, root_model, original_model, device)
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
        embed_metrics = evaluate_model(dataloader, embed_model, original_model, device)
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
    svd_metrics = evaluate_model(dataloader, svd_model, original_model, device)
    print(f"SVD baseline experiment, rank={svd_rank}")
    print("-----------------")
    for key, value in svd_metrics.items():
        print(f"{key}: {value}")
