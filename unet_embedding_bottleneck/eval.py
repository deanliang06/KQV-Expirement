from pathlib import Path

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from gpt_new_attn_matrix import GPTAttn
from unet_embedding_auto import UNetEmbeddingAutoencoder


SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = SCRIPT_DIR / "embedding_auto_model.pth"


def encode(example):
    return tok(example["text"])


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


def test(dataloader, autoencoder_model, original_model):
    original_correct = 0
    my_correct = 0

    print(f"Num batches in test: {len(dataloader)}")
    with torch.no_grad():
        for x in dataloader:
            last_logit = x["input_ids"][:, -1].to("cpu")
            for k in x:
                x[k] = x[k][:, :-1].to("cuda", non_blocking=True)

            actual_y = original_model(**x).logits[:, -1:, :]
            pred = autoencoder_model(**x).logits[:, -1:, :]

            actual_logit = actual_y.argmax(-1).to("cpu")
            my_logit = pred.argmax(-1).to("cpu")

            my_correct += (my_logit == last_logit).to(torch.float16).mean()
            original_correct += (actual_logit == last_logit).to(torch.float16).mean()

    print("LAMBADA eval\n-----------------")
    print(f"Original distilgpt2 accuracy: {100 * original_correct / len(dataloader)}")
    print(f"Embedding bottleneck model accuracy: {100 * my_correct / len(dataloader)}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=data_collator,
    )

    original_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).eval()
    autoencoder_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).eval()
    model = UNetEmbeddingAutoencoder(768, 128).to(device)

    if CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    else:
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT_PATH}")

    attach_gpt_wrappers(autoencoder_model, original_model, model, device=device)
    test(test_dataloader, autoencoder_model, original_model)
