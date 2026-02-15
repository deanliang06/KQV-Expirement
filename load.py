from transformers import AutoModel, AutoTokenizer
import torch
import sys
from new_attention_module import FragmentedAttentionModule

if __name__ == "__main__":
    url = "huawei-noah/TinyBERT_General_4L_312D"
    tok = AutoTokenizer.from_pretrained(url)
    model = AutoModel.from_pretrained(url, device_map='auto')

    auto_encoder = FragmentedAttentionModule().to("cuda")

    for layer in model.encoder.layer:
        for param in layer.attention.self.query.parameters():
            print(auto_encoder(param))
            sys.exit()
