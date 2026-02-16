from transformers import AutoModel, AutoTokenizer
import torch
import sys
from new_attention_module import Autoencoder, FragmentedKQV

if __name__ == "__main__":
    url = "huawei-noah/TinyBERT_General_4L_312D"
    tok = AutoTokenizer.from_pretrained(url)
    model = AutoModel.from_pretrained(url, device_map='auto')

    auto_encoder = Autoencoder().to("cuda")
    model_inputs = tok(["The secret to baking a good cake is "], return_tensors="pt").to(model.device)
    original_output = model(**model_inputs)
    print("Original Model: ")
    print(original_output.last_hidden_state[:,0,:])

    for layer in model.encoder.layer:
        newModel = {}
        need = False
        for param in layer.attention.self.query.parameters():
            need = True
            if (param.dim() == 1):
                newModel["bias"] = param
                continue
            comp, decomp, feature = auto_encoder(param)
            newModel["comp"] = comp
            newModel["decomp"] = decomp
            newModel["feature"] = feature
        
        if need:
            layer.attention.self.query = FragmentedKQV(newModel, d=64)
            need = False

        for param in layer.attention.self.key.parameters():
            need = True
            if (param.dim() == 1):
                newModel["bias"] = param
                continue
            comp, decomp, feature = auto_encoder(param)
            newModel["comp"] = comp
            newModel["decomp"] = decomp
            newModel["feature"] = feature

        if need:
            layer.attention.self.key = FragmentedKQV(newModel, d=64)
            need = False
        
        for param in layer.attention.self.value.parameters():
            need = True
            if (param.dim() == 1):
                newModel["bias"] = param
                continue
            comp, decomp, feature = auto_encoder(param)
            newModel["comp"] = comp
            newModel["decomp"] = decomp
            newModel["feature"] = feature
        
        if need:
            layer.attention.self.value = FragmentedKQV(newModel, d=64)
            need = False
        
    model_inputs = tok(["The secret to baking a good cake is "], return_tensors="pt").to(model.device)
    generated_ids = model(**model_inputs)
    print("Autoencoder reconstructed model: ")
    print(generated_ids.last_hidden_state[:,0,:])
            
