import torch
import torch.nn as nn
import sys

class FragmentedAttentionModule(nn.Module):
    def __init__(self, d = 120):
        super().__init__()
        self.tokenizer = nn.Linear(312,64)
        self.attention = nn.MultiheadAttention(64, 4, 0.05, batch_first=True)
        self.attentionTwo = nn.MultiheadAttention(64, 4, 0.05, batch_first=True)
        self.summary_token = nn.Parameter(torch.randn(64))
        self.attentionThree = nn.MultiheadAttention(64, 4, 0.05, batch_first=True)

        self.proj = nn.Linear(64, 312)
        self.attentionFour = nn.MultiheadAttention(64, 4)
        self.attentionFive = nn.MultiheadAttention(64, 4)
        self.detokenizer = nn.Linear(64, 312)
    def forward(self, x):
        #encoding
        x = self.tokenizer(x)
        attn,_ = self.attention(x,x,x)
        attn_two,_ = self.attentionTwo(attn,attn,attn)
        final = torch.concatenate((torch.unsqueeze(self.summary_token, 0),attn_two))
        final_enc,_ = self.attentionThree(final, final, final)
        hyper_rep = final_enc[0]
        #decoding
        proj = self.proj(hyper_rep)
        attn_three,_ = self.attentionFour(proj, proj, proj)
        attn_four,_ = self.attentionFive(attn_three, attn_three, attn_three)
        final = self.detokenizer(attn_four)
        return final


