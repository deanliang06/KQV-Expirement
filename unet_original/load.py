from contextlib import nullcontext
from pathlib import Path
import os
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
    from unet_original.gpt_new_attn_matrix import GPTAttn
    from unet_original.unet_auto import CNNAutoencoder
else:
    from .gpt_new_attn_matrix import GPTAttn
    from .unet_auto import CNNAutoencoder


TOKEN_CE_WEIGHT = 1.0
DISTILL_KL_WEIGHT = 0.8
DISTILL_TEMPERATURE = 2.0
CHECKPOINT_PATH = Path(__file__).with_name("auto_encoder.pth")


def encode(example):
    return tok(example["text"])


def decode(text):
    return tok.decode(text, skip_special_tokens=True)


def distillation_loss(teacher_logits, student_logits, targets, ignore_index, kl_weight=DISTILL_KL_WEIGHT, temperature=DISTILL_TEMPERATURE):
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)

    vocab = student_logits.size(-1)
    ce_loss = F.cross_entropy(student_logits.reshape(-1, vocab), targets.reshape(-1), ignore_index=ignore_index)
    total = TOKEN_CE_WEIGHT * ce_loss + kl_weight * kl_loss
    return total, ce_loss.detach(), kl_loss.detach()


def attach_gpt_wrappers(target_model, source_model, autoencoder, d=128, device="cuda"):
    for i, layer in enumerate(source_model.transformer.h):
        params = {}
        for name, param in layer.attn.c_attn.named_parameters():
            if name == "bias":
                params["bias"] = param
            else:
                params["weights"] = param

        if not isinstance(target_model.transformer.h[i].attn.c_attn, GPTAttn):
            target_model.transformer.h[i].attn.c_attn = GPTAttn(autoencoder, params, d=d).to(device)
        else:
            target_model.transformer.h[i].attn.c_attn.set_autoencoder(autoencoder)


def collect_trainable_params(target_model, autoencoder):
    trainable = []
    seen = set()

    def add_param(param):
        if not param.requires_grad:
            return
        pid = id(param)
        if pid in seen:
            return
        seen.add(pid)
        trainable.append(param)

    for param in autoencoder.parameters():
        add_param(param)

    for layer in target_model.transformer.h:
        wrapped_attn = layer.attn.c_attn
        if not isinstance(wrapped_attn, GPTAttn):
            continue

        for module in (
            wrapped_attn.UNet_layer.down,
            wrapped_attn.UNet_layer.down_ll,
            wrapped_attn.UNet_layer.up,
            wrapped_attn.UNet_layer.up_ll,
        ):
            for param in module.parameters():
                add_param(param)

    return trainable


def autocast_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def move_batch_to_device(batch, device):
    for key in batch:
        batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
    return batch


def test(dataloader, autoencoder_model, original_model, epoch_num, perplexity, device):
    total_loss = 0
    total_ce_loss = 0
    total_kl_loss = 0
    cos_sim = 0
    total_perplexity = 0

    with torch.no_grad():
        for x in dataloader:
            x = move_batch_to_device(x, device)

            with autocast_context(device):
                y = original_model(**x).logits[:, :-1, :]
                pred = autoencoder_model(**x).logits[:, :-1, :]
                target = x["input_ids"][:, 1:]
                loss, ce_loss, kl_loss = distillation_loss(y, pred, target, tok.pad_token_id)

                if not torch.isfinite(loss):
                    continue

            pscore = perplexity(preds=pred, target=target)
            if not torch.isfinite(pscore):
                continue

            total_loss += loss.detach()
            total_ce_loss += ce_loss
            total_kl_loss += kl_loss
            cos = torch.nn.functional.cosine_similarity(pred, y, dim=-1).mean()
            cos_sim += cos.item()
            total_perplexity += pscore.item() / len(dataloader)

    print(f"Epoch Test #{epoch_num}\n-----------------")
    print(f"Average Perplexity: {total_perplexity}")
    print(f"Distill Loss: {total_loss / len(dataloader)}")
    print(f"Token CE: {total_ce_loss / len(dataloader)}")
    print(f"KL Loss: {total_kl_loss / len(dataloader)}")
    print(f"Cos Sim: {cos_sim / len(dataloader)}")


def train(dataloader, autoencoder_model, original_model, autoencoder, optimizer, epoch_num, scalar, device, perplexity=None):
    total_loss = 0
    total_ce_loss = 0
    total_kl_loss = 0
    cos_sim = 0
    total_perplexity = 0
    print(f"Num batches: {len(dataloader)}")
    for x in dataloader:
        x = move_batch_to_device(x, device)
        with torch.no_grad():
            actual_y = original_model(**x).logits[:, :-1, :]

        with autocast_context(device):
            pred = autoencoder_model(**x).logits[:, :-1, :]

        pred = pred.to(torch.float32)
        target = x["input_ids"][:, 1:]
        loss, ce_loss, kl_loss = distillation_loss(actual_y, pred, target, tok.pad_token_id)
        scalar.scale(loss).backward()
        scalar.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)

        if not torch.isfinite(grad_norm):
            for name, p in autoencoder.named_parameters():
                if p.grad is None:
                    continue
                if p.grad.nan_to_num(0).mean() == 0:
                    print(f"{name} is problematic")
            optimizer.zero_grad(set_to_none=True)
            scalar.update()
            continue

        scalar.step(optimizer)
        scalar.update()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            pnum = perplexity(preds=pred, target=target)
            if not torch.isfinite(pnum):
                print("Perplexity is not finite")
                continue
            total_perplexity += pnum.item() / len(dataloader)
            total_ce_loss += ce_loss
            total_kl_loss += kl_loss
            cos = torch.nn.functional.cosine_similarity(pred, actual_y, dim=-1).mean()
            cos_sim += cos.item()
            total_loss += loss.detach()

    print(f"Epoch Train #{epoch_num}\n-----------------")
    print(f"Perplexity: {total_perplexity}")
    print(f"Distill Loss: {total_loss / len(dataloader)}")
    print(f"Token CE: {total_ce_loss / len(dataloader)}")
    print(f"KL Loss: {total_kl_loss / len(dataloader)}")
    print(f"Cos Sim: {cos_sim / len(dataloader)}")


if __name__ == "__main__":
    lr = 3e-5
    wd = 0.002
    epochs = 10
    n_embed = 768

    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    url = "distilgpt2"

    tok = AutoTokenizer.from_pretrained(url)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(2, os.cpu_count() or 0)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, return_tensors="pt")
    model = AutoModelForCausalLM.from_pretrained(url).to(device).eval()
    dataset = train_dataset.map(encode, batched=True, batch_size=1000, remove_columns=["text"]).with_format(type="torch")
    t_dataset = test_dataset.map(encode, batched=True, batch_size=1000, remove_columns=["text"]).with_format(type="torch")
    data = DataLoader(
        dataset,
        batch_size=12,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=data_collator,
    )
    t_data = DataLoader(
        t_dataset,
        batch_size=12,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=data_collator,
    )

    changed_model = AutoModelForCausalLM.from_pretrained(url).to(device).eval()
    for p in changed_model.parameters():
        p.requires_grad_(False)

    auto_encoder = CNNAutoencoder(ndim=n_embed, d=128).to(device)
    if CHECKPOINT_PATH.exists():
        auto_encoder.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    attach_gpt_wrappers(changed_model, model, auto_encoder, d=128, device=device)
    optimizer = torch.optim.Adam(collect_trainable_params(changed_model, auto_encoder), lr, weight_decay=wd)
    scalar = torch.amp.GradScaler(enabled=device.type == "cuda")
    perplexity = Perplexity(ignore_index=tok.pad_token_id).to(device)
    for it in range(epochs):
        train(data, changed_model, model, auto_encoder, optimizer, it + 1, scalar, device, perplexity)
        test(t_data, changed_model, model, it + 1, perplexity, device)
        torch.save(auto_encoder.state_dict(), CHECKPOINT_PATH)
