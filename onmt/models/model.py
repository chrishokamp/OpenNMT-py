""" Onmt NMT Model base class definition """

import torch.nn as nn

from onmt.attention_bridge import AttentionBridge


class MultiTaskModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self,
                 encoder_map,
                 decoder_map,
                 generator_map,
                 init_decoder='rnn_final_state',
                 bridge=None,
                 multigpu=False):

        self.multigpu = multigpu
        super(MultiTaskModel, self).__init__()

        # we need to transform the maps into id maps and module lists so that
        # the components correctly get registered as pytorch modules
        self.encoder_ids = {lang_code: idx
                            for lang_code, idx
                            in zip(encoder_map.keys(),
                                   range(len(encoder_map)))}
        self.encoders = nn.ModuleList(encoder_map.values())

        self.decoder_ids = {lang_code: idx
                            for lang_code, idx
                            in zip(decoder_map.keys(),
                                   range(len(decoder_map)))}
        self.decoders = nn.ModuleList(decoder_map.values())

        self.generator_ids = {lang_code: idx
                              for lang_code, idx
                              in zip(decoder_map.keys(),
                                     range(len(decoder_map)))}
        self.generators = nn.ModuleList(generator_map.values())

        assert (len(self.decoders) == len(self.generators),
                'Generators are linked to decoders.')

        self.bridge = None
        if bridge is not None:
            self.bridge = bridge

        assert init_decoder in ['rnn_final_state', 'attention_matrix']
        self.init_decoder = init_decoder

    def forward(self, src, tgt, src_task, tgt_task, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        encoder = self.encoders[self.encoder_ids[src_task]]
        decoder = self.decoders[self.decoder_ids[tgt_task]]

        enc_final, memory_bank = encoder(src, lengths)

        # Implement attention bridge/compound attention
        if self.bridge is not None:
            enc_final, memory_bank = self.bridge(memory_bank)
        
        # initialize decoder
        # Note: this assert should probably be in initialization
        if str(type(encoder)).find('transformer.TransformerEncoder') > -1:
            assert (self.init_decoder == "attention_matrix") , \
               ("""Unsupported decoder initialization {}. Use 
                the 'attention matrix' option for the '-init_decoder'
                flag when using a transformer encoder""".format(
                   self.init_decoder))

        if self.init_decoder == 'attention_matrix':
            enc_state = \
                self.attention_bridge.init_decoder_state(src,
                                                         memory_bank,
                                                         enc_final)
        else:
            enc_state = decoder.init_decoder_state(
                src,
                memory_bank,
                enc_final)

        decoder_outputs, dec_state, attns = \
            decoder(tgt, memory_bank,
                    enc_state if dec_state is None else dec_state,
                    memory_lengths=lengths)

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None

        return decoder_outputs, attns, dec_state


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder, model_opt, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.use_attention_bridge = model_opt.use_attention_bridge
        self.attention_bridge = AttentionBridge(model_opt.rnn_size, 
                                                    model_opt.attention_heads, 
                                                    model_opt.dec_layers) #, model_opt.dropout)
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_final, memory_bank = self.encoder(src, lengths)
        enc_state = \
            self.decoder.init_decoder_state(src, memory_bank, enc_final)
        # Implement attention bridge/compound attention
        if self.use_attention_bridge:
            enc_final, memory_bank = self.attention_bridge(memory_bank)

        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state
