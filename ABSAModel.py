# -*- coding: utf-8 -*-
"""
Contains the AbsaGRU model.

@author: Geerthan
"""

import torch.nn as nn
import torch

class AbsaGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_units, batch_size, out_dim, device, init_embed):
        super(AbsaGRU, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.out_dim = out_dim
        self.device = device

        self.embed = torch.nn.Embedding.from_pretrained(init_embed, freeze=False)
        self.dropout = nn.Dropout(p=0.3)
        self.gru = nn.GRU(self.embed_dim, self.hidden_units, batch_first=True, bidirectional=True)
        self.att = nn.MultiheadAttention(self.hidden_units*2, 1, batch_first=True) # num_heads = 1
        self.norm = nn.BatchNorm1d(228)
        self.fc = nn.Linear(self.hidden_units*2 + 128, self.out_dim)
        
        self.noneTensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device)
        self.noneTensor2D = self.noneTensor.repeat(2,1)
        
    def forward(self, ids, lbls, subs, sents):
        # Turn words into embeddings
        ids = self.embed(ids)
        ids = self.dropout(ids)
        
        # Needed for indexing later
        batch = ids.shape[0]
        first = torch.arange(0, batch, dtype=torch.long)

        # GRU + attention
        out, self.hidden = self.gru(ids)
        out_att, out_weight = self.att(out, out, out)
        
        # Get output corresponding to desired word location
        sb = subs[:, 0]
        out_att = out_att[first, sb, :]
        out_weight = out_weight[first, sb, :]
        
        # Weight sentiments
        weighted_sents = sents * out_weight
                
        # Dropout + fuse features
        out_att = self.dropout(out_att)
        out_sents = self.dropout(weighted_sents)
        out_all = torch.cat((out_att, out_sents), dim=1)
        
        # Normalize + Linear
        out_all = self.norm(out_all)
        out_all = self.fc(out_all)
        
        return out_all, self.hidden
    