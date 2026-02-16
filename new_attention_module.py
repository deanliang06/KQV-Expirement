import torch
import torch.nn as nn
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
    def __init__(self, params, d = 64):
        super().__init__()
        self.comp = nn.Linear(312,d, bias=False)
        self.comp.weight.data = params["comp"].reshape(64, 312)

        self.feature = nn.Linear(d,d, bias=False)
        self.feature.weight.data = params["feature"].reshape(d, d)

        self.decomp = nn.Linear(d, 312)
        self.decomp.weight.data = params["decomp"].reshape(312, d)
        self.decomp.bias.data = params["bias"]
    
    def forward(self, x):
        return self.decomp(self.feature(self.comp(x)))


