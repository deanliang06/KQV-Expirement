import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class Autoencoder(nn.Module):
    def __init__(self, d = 64):
        super().__init__()
        self.d = d
        self.tokenizer = nn.Linear(312,64)
        self.attention = nn.MultiheadAttention(64, 4, 0.05, batch_first=True)
        self.attentionTwo = nn.MultiheadAttention(64, 4, 0.05, batch_first=True)
        self.summary_token = nn.Parameter(torch.randn(64))
        self.attentionThree = nn.MultiheadAttention(64, 4, 0.05, batch_first=True)

        self.proj = nn.Linear(64, 2*312*64+d*64)
        self.attentionFour = nn.MultiheadAttention(64, 4)
        self.attentionFive = nn.MultiheadAttention(64, 4)
        self.detokenizer_compress = nn.Linear(64, d)
        self.detokenizer_decompress = nn.Linear(64, d)
        self.feature_detoken = nn.Linear(64, d)

    def forward(self, x):
        #encoding
        x = self.tokenizer(x)
        attn,_ = self.attention(x,x,x)
        attn_two,_ = self.attentionTwo(attn,attn,attn)
        final = torch.concatenate((torch.unsqueeze(self.summary_token, 0),attn_two))
        final_enc,_ = self.attentionThree(final, final, final)
        hyper_rep = final_enc[0]

        #decoding
        proj = self.proj(hyper_rep).reshape(2*312+self.d, 64)
        attn_three,_ = self.attentionFour(proj, proj, proj)
        attn_four,_ = self.attentionFive(attn_three, attn_three, attn_three)
        
        split = torch.split(attn_four, [312, 312, self.d])
        #THIS IS THE ISSUE
        return self.detokenizer_compress(split[0]), self.detokenizer_decompress(split[1]), self.feature_detoken(split[2])
    
class FragmentedKQV(nn.Module):
    def __init__(self, autoencoder, params, d = 64):
        super().__init__()
        self.d = d
        self.autoencoder = autoencoder
        self.base_weights = params["weights"].detach()
        self.base_bias = params["bias"].detach()

    def forward(self, x):
        comp, decomp, feature = self.autoencoder(self.base_weights)
        comp = comp.reshape(self.d, 312)
        decomp = decomp.reshape(312, self.d)
        feature = feature.reshape(self.d, self.d)
        x = F.linear(x, comp)
        x = F.linear(x, feature)
        return F.linear(x, decomp, self.base_bias)
    


