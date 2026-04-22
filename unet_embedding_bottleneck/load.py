from contextlib import nullcontext
from pathlib import Path
import os
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
    from unet_embedding_bottleneck.gpt_new_attn_matrix import GPTAttn
    from unet_embedding_bottleneck.unet_embedding_auto import UNetEmbeddingAutoencoder
else:
    from .gpt_new_attn_matrix import GPTAttn
    from .unet_embedding_auto import UNetEmbeddingAutoencoder


SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = SCRIPT_DIR / "embedding_auto_model.pth"
MATRIX_L2_WEIGHT = 1.0
TOKEN_CE_WEIGHT = 0.1


def encode(example):
    return tok(example["text"])


def matrix_l2_loss(model):
    total = 0.0
    wrapped_layers = 0

    for layer in model.transformer.h:
        wrapped_attn = layer.attn.c_attn
        if not isinstance(wrapped_attn, GPTAttn):
            continue

        recon_q = wrapped_attn._cached_q_weight
        if recon_q is None:
            recon_q = wrapped_attn.reconstruct_q()
        total = total + (recon_q - wrapped_attn.q).pow(2).mean()
        wrapped_layers += 1

    if wrapped_layers == 0:
        raise RuntimeError("No embedding bottleneck GPTAttn wrappers found")

    return total / wrapped_layers


def clear_q_weight_caches(model):
    for layer in model.transformer.h:
        wrapped_attn = layer.attn.c_attn
        if isinstance(wrapped_attn, GPTAttn):
            wrapped_attn.clear_cache()


def token_cross_entropy_loss(logits, targets, ignore_index):
    vocab = logits.size(-1)
    return F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), ignore_index=ignore_index)


def combined_loss(model, pred, targets, ignore_index):
    matrix_loss = matrix_l2_loss(model)
    ce_loss = token_cross_entropy_loss(pred, targets, ignore_index)
    total = MATRIX_L2_WEIGHT * matrix_loss + TOKEN_CE_WEIGHT * ce_loss
    return total, matrix_loss.detach(), ce_loss.detach()


def attach_gpt_wrappers(target_model, source_model, autoencoder, device="cuda"):
    for i, layer in enumerate(source_model.transformer.h):
        params = {}
        for name, param in layer.attn.c_attn.named_parameters():
            if name == "bias":
                params["bias"] = param
            else:
                params["weights"] = param

        if not isinstance(target_model.transformer.h[i].attn.c_attn, GPTAttn):
            target_model.transformer.h[i].attn.c_attn = GPTAttn(autoencoder, params).to(device)
        else:
            target_model.transformer.h[i].attn.c_attn.set_autoencoder(autoencoder)


def collect_trainable_params(autoencoder):
    return [param for param in autoencoder.parameters() if param.requires_grad]


def autocast_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def move_batch_to_device(batch, device):
    for key in batch:
        batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")
    return batch


def test(dataloader, autoencoder_model, original_model, epoch_num, device):
    total_loss = 0
    total_matrix_l2 = 0
    total_token_ce = 0
    total_mse = 0

    with torch.no_grad():
        for x in dataloader:
            x = move_batch_to_device(x, device)

            with autocast_context(device):
                y = original_model(**x).logits[:, :-1, :]
                pred = autoencoder_model(**x).logits[:, :-1, :]
                target = x["input_ids"][:, 1:]
                loss, matrix_l2, token_ce = combined_loss(autoencoder_model, pred, target, tok.pad_token_id)
                mse = F.mse_loss(pred.float(), y.float())

                if not torch.isfinite(loss) or not torch.isfinite(mse):
                    clear_q_weight_caches(autoencoder_model)
                    continue

            total_loss += loss.detach()
            total_matrix_l2 += matrix_l2
            total_token_ce += token_ce
            total_mse += mse.item()
            clear_q_weight_caches(autoencoder_model)

    print(f"Epoch Test #{epoch_num}\n-----------------")
    print(f"Combined Loss: {total_loss / len(dataloader)}")
    print(f"Matrix L2: {total_matrix_l2 / len(dataloader)}")
    print(f"Token CE: {total_token_ce / len(dataloader)}")
    print(f"MSE: {total_mse / len(dataloader)}")


def train(dataloader, autoencoder_model, original_model, autoencoder, optimizer, epoch_num, scalar, device):
    total_loss = 0
    total_matrix_l2 = 0
    total_token_ce = 0
    total_mse = 0
    print(f"Num batches: {len(dataloader)}")

    for x in dataloader:
        x = move_batch_to_device(x, device)

        with torch.no_grad():
            with autocast_context(device):
                actual_y = original_model(**x).logits[:, :-1, :]

        with autocast_context(device):
            pred = autoencoder_model(**x).logits[:, :-1, :]
            target = x["input_ids"][:, 1:]
            loss, matrix_l2, token_ce = combined_loss(autoencoder_model, pred, target, tok.pad_token_id)
        scalar.scale(loss).backward()
        scalar.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)

        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            scalar.update()
            clear_q_weight_caches(autoencoder_model)
            continue

        scalar.step(optimizer)
        scalar.update()
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            mse = F.mse_loss(pred.float(), actual_y.float())
            if not torch.isfinite(mse):
                clear_q_weight_caches(autoencoder_model)
                continue
            total_matrix_l2 += matrix_l2
            total_token_ce += token_ce
            total_mse += mse.item()
            total_loss += loss.detach()
        clear_q_weight_caches(autoencoder_model)

    print(f"Epoch Train #{epoch_num}\n-----------------")
    print(f"Combined Loss: {total_loss / len(dataloader)}")
    print(f"Matrix L2: {total_matrix_l2 / len(dataloader)}")
    print(f"Token CE: {total_token_ce / len(dataloader)}")
    print(f"MSE: {total_mse / len(dataloader)}")


if __name__ == "__main__":
    lr = 3e-5
    wd = 0.002
    epochs = 75
    n_embed = 768
    embedding_dim = 128

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

    auto_encoder = UNetEmbeddingAutoencoder(ndim=n_embed, embedding_dim=embedding_dim).to(device)
    if CHECKPOINT_PATH.exists():
        auto_encoder.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))

    attach_gpt_wrappers(changed_model, model, auto_encoder, device=device)
    optimizer = torch.optim.Adam(collect_trainable_params(auto_encoder), lr, weight_decay=wd)
    scalar = torch.amp.GradScaler(enabled=device.type == "cuda")

    for it in range(epochs):
        train(data, changed_model, model, auto_encoder, optimizer, it + 1, scalar, device)
        test(t_data, changed_model, model, it + 1, device)
        torch.save(auto_encoder.state_dict(), CHECKPOINT_PATH)
