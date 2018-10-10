"""Multi-headed attention"""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

#from torch.nn.utils.rnn import pack_padded_sequence as pack
#from torch.nn.utils.rnn import pad_packed_sequence as unpack

#from onmt.encoders.encoder import EncoderBase
#from onmt.utils.rnn_factory import *
from onmt.decoders.decoder import RNNDecoderState
#from onmt.decoders import decoder


class AttentionBridge(nn.Module):
    """
    Multi-headed attention. Bridge between encoders->decoders
    """
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 dec_num_layers,
                 bridge_type='matrix',
                 pooling=None,
                 dropout=0.05):

        """
        If bridge_type is 'matrix', the intermediate representation follows
          Lin et al 2017. If it's 'feed-forward', the bridge is a feed-forward
          mapping of the encoder states.

        """
        super(AttentionBridge, self).__init__()

        u = hidden_size
        d = hidden_size
        r = num_attention_heads
        self.drop = nn.Dropout(dropout)
        self.ws1 = nn.Linear(d, u, bias=False)
        self.ws2 = nn.Linear(d, r, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.attention_hops = r
        self.hidden_size = hidden_size
        self.dec_layers = dec_num_layers
        self.num_attention_heads = num_attention_heads

        self.W_init = nn.Linear(u,dec_num_layers*u , bias=False)
        self.tanh = nn.Tanh()

    # this currently misses the multi-head attention distribution
    #   regularization term from Lin et al
    def forward(self, enc_output):
        # TODO: implement different bridge types
        output, alphas = self.mix_attention(enc_output)
        # take transpose to match dimensions s.t. r=new_seq_len
        M = torch.transpose(output, 0, 1).contiguous() #[r,bsz,nhid]

        # Chris: this is the pooling operation
        h_avrg = self.pool(M)

        return h_avrg, M

    def pool(self, states, op='mean'):
        """
        Pools the states along the time dimension into a single vector
        """

        # Chris: this may result in a decoder initialization bug -- should not
        # Chris: change the first dim of the encoder states
        if op == 'mean':
            return states.mean(dim=0, keepdim=True)
        else:
            raise ValueError('unknown pooling op: {}'.format(op))

    # TODO: debug to match Lin et al -- softmax looks wrong
    def mix_attention(self, output):
        """
        Notation based on Lin et al. (2017)
        A structured self-attentive sentence embedding"""
        outp = torch.transpose(output, 0, 1).contiguous()
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]
        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        alphas = alphas.view(-1, size[1]) # [bsz*hop, len]
        alphas = self.softmax(alphas)  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

    def init_decoder_state(self, src, memory_bank, encoder_final):
        '''
        initialize the decoder state, `s_0`.
        We use a similar s_0 as in rSennrich et al.(2017) "Nematus":
             - s_0 = tanh(W_init * h_avrg);
                  where h_avrg is the average of the sentence embedding, M,
                  (instead of taking the average over the hidden states of
                   the RNN, as in rSennrich(2017) over the )
        '''
        s_0 = self.tanh(self.W_init(encoder_final))
        ss = s_0[:,:,:self.hidden_size]
        for i in range(self.dec_layers - 1):
            init = (i+1) * self.hidden_size
            end = (i+2) * self.hidden_size
            ss = torch.cat([ss,s_0[:,:,init:end]])
        s_0 = ss
        return RNNDecoderState(self.hidden_size, s_0)

