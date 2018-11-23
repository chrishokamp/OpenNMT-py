"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

import onmt
from onmt.encoders.encoder import EncoderBase
# from onmt.utils.misc import aeq
from onmt.modules.position_ffn import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 self_attn=None, cache_weights=False):
        super(TransformerEncoderLayer, self).__init__()

        # if a self_attn wasn't provided, init one for this layer
        if self_attn is None:
            self.self_attn = onmt.modules.MultiHeadedAttention(
                heads, d_model, dropout=dropout)
        else:
            self.self_attn = self_attn

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = onmt.modules.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.cache_weights = cache_weights
        self.cache = {}

    def forward(self, inputs, mask, mask_heads_after=None):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        context, attn = self.self_attn(input_norm, input_norm, input_norm,
                                       mask=mask,
                                       mask_heads_after=mask_heads_after)

        # attach attention weights to cache
        if self.cache_weights:
            self.cache['attention_weights'] = attn
            # import ipdb;ipdb.set_trace()

        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings,
                 cache_weight_layers=None,
                 share_self_attn=False):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
        if cache_weight_layers is None:
            cache_weight_layers = []

        self_attn = None
        if share_self_attn:
            self_attn = onmt.modules.MultiHeadedAttention(
                heads, d_model, dropout=dropout)

        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout,
                self_attn=self_attn,
                cache_weights=True if layer_idx in cache_weight_layers
                else False)
             for layer_idx in range(num_layers)])
        self.layer_norm = onmt.modules.LayerNorm(d_model)

        # this can be set dynamically externally
        self.num_heads = heads
        self.mask_heads_after = None

    def forward(self, src, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask,
                                      mask_heads_after=self.mask_heads_after)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths
