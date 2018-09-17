"""Multi-headed attention"""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

#from torch.nn.utils.rnn import pack_padded_sequence as pack
#from torch.nn.utils.rnn import pad_packed_sequence as unpack

#from onmt.encoders.encoder import EncoderBase
#from onmt.utils.rnn_factory import rnn_factory


class AttentionBridge(nn.Module):
    """
    Multi-headed attention. Bridge between encoders->decoders
    """
    def __init__(self, hidden_size, attention_heads, dropout=0.05):
        """Attention Heads Layer:"""
        super(AttentionBridge, self).__init__()
        u = hidden_size
        d = hidden_size
        r = attention_heads
        self.drop = nn.Dropout(dropout)
        self.ws1 = nn.Linear(d, u, bias=False)
        self.ws2 = nn.Linear(d, r, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        #        self.init_weights()
        self.attention_hops = r
        self.M = None

    def forward(self, enc_output):
        output, alphas = self.mixAtt(enc_output)
        # take transpose to match dimensions s.t. r=new_seq_len
        self.M = torch.transpose(output, 0, 1).contiguous() #[r,bsz,nhid]
        h_avrg = (self.M).mean(dim=0, keepdim=True)

        return h_avrg, self.M # enc_final=h_avrg memory_bank=output3

    def mixAtt(self, outp):
        """Notation based on Lin et al. (2017) A structured self-attentive sentence embedding"""
        outp = torch.transpose(outp, 0, 1).contiguous()
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]
        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        alphas = alphas.view(-1, size[1]) # [bsz*hop, len]
        alphas = self.softmax(alphas)  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas
