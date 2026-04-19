import argparse

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from gpt_new_attn_matrix import GPTAttn


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


def test(dataloader, approx_model, original_model, device):
    original_correct = 0
    svd_correct = 0

    print(f"Num batches in test: {len(dataloader)}")
    with torch.no_grad():
        for x in dataloader:
            last_logit = x["input_ids"][:, -1].to("cpu")
            for k in x:
                x[k] = x[k][:, :-1].to(device, non_blocking=True)

            actual_y = original_model(**x).logits[:, -1:, :]
            pred = approx_model(**x).logits[:, -1:, :]

            actual_logit = actual_y.argmax(-1).to("cpu")
            svd_logit = pred.argmax(-1).to("cpu")

            svd_correct += (svd_logit == last_logit).to(torch.float16).mean()
            original_correct += (actual_logit == last_logit).to(torch.float16).mean()

    print("LAMBADA eval\n-----------------")
    print(f"Original distilgpt2 accuracy: {100 * original_correct / len(dataloader)}")
    print(f"SVD baseline accuracy: {100 * svd_correct / len(dataloader)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = load_dataset("cimec/lambada", split="test")

    tok = AutoTokenizer.from_pretrained("distilgpt2", max_length=512)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, return_tensors="pt")
    test_dataset = test_dataset.map(encode, batched=True, batch_size=1000, remove_columns=["text", "domain"]).with_format(type="torch")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=data_collator,
    )

    original_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).eval()
    svd_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).eval()
    attach_gpt_wrappers(svd_model, original_model, rank=args.rank, device=device)
    test(test_dataloader, svd_model, original_model, device)
