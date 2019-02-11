"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters
import onmt.modules
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.audio_encoder import AudioEncoder
from onmt.encoders.image_encoder import ImageEncoder
from onmt.encoders import str2enc

from onmt.decoders.decoder import InputFeedRNNDecoder, StdRNNDecoder
from onmt.decoders.transformer import TransformerDecoder
from onmt.decoders.cnn_decoder import CNNDecoder
from onmt.modules.attention_bridge import AttentionBridge
from onmt.decoders import str2dec

from onmt.modules import Embeddings, CopyGenerator
from onmt.modules.util_class import Cast
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger


def build_embeddings(opt, word_dict, feature_dicts, for_encoder=True,
                     pad_word='<blank>'):

    """
    Build an Embeddings instance.
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[pad_word]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[pad_word]
                         for feat_dict in feature_dicts]

    num_feat_embeddings = [len(feat_dict) for feat_dict in
                          feature_dicts]

    #pad_indices = [f.vocab.stoi[f.pad_token] for _, f in text_field]
    #word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]

    #num_embs = [len(f.vocab) for _, f in text_field]
    #num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

    fix_word_vecs = opt.fix_word_vecs_enc if for_encoder \
        else opt.fix_word_vecs_dec

    return Embeddings(
        word_vec_size=embedding_dim,
        position_encoding=opt.position_encoding,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feats_padding_idx,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam",
        fix_word_vecs=fix_word_vecs
    )
    return emb


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    if opt.encoder_type == "transformer":
        return TransformerEncoder(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout,
            embeddings,
            cache_weight_layers=getattr(opt, 'enc_cache_weight_layers', None),
            share_self_attn=getattr(opt, 'enc_share_self_attn', False))
    elif opt.encoder_type == "cnn":
        encoder = CNNEncoder(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.cnn_kernel_width,
            opt.dropout,
            embeddings)
    elif opt.encoder_type == "mean":
        encoder = MeanEncoder(opt.enc_layers, embeddings)
    else:
        encoder = RNNEncoder(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout,
            embeddings,
            opt.bridge
        )
    return encoder

    #enc_type = opt.encoder_type if opt.model_type == "text" else opt.model_type
    #return str2enc[enc_type].from_opt(opt, embeddings)


def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if opt.decoder_type == "transformer":
        decoder = TransformerDecoder(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.global_attention,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout,
            embeddings,
            cache_weight_layers=getattr(opt, 'dec_cache_weight_layers', None))
    elif opt.decoder_type == "cnn":
        decoder = CNNDecoder(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.copy_attn,
            opt.cnn_kernel_width,
            opt.dropout,
            embeddings
        )
    else:
        dec_class = InputFeedRNNDecoder if opt.input_feed else StdRNNDecoder
        decoder = dec_class(
            opt.rnn_type,
            opt.brnn,
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.global_attention_function,
            opt.coverage_attn,
            opt.context_gate,
            opt.copy_attn,
            opt.dropout,
            embeddings,
            opt.reuse_copy_attn
        )
    return decoder
    #dec_type = "ifrnn" if opt.decoder_type == "rnn" and opt.input_feed \
    #           else opt.decoder_type
    #return str2dec[dec_type].from_opt(opt, embeddings)


def load_test_multitask_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)
    model = checkpoint['whole_model']
    device = torch.device("cuda" if use_gpu(opt) else "cpu")
    model.to(device)
    model.eval()
    return model


def load_test_model(opt, dummy_opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    model_opt = checkpoint['opt']
    vocab = checkpoint['vocab']
    if inputters.old_style_vocab(vocab):
        fields = inputters.load_old_vocab(
            vocab, opt.data_type, dynamic_dict=model_opt.copy_attn
        )
    else:
        fields = vocab

    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def build_embeddings_then_encoder(model_opt, fields):
    if model_opt.model_type == "text":
        src_dict = fields["src"].vocab
        feature_dicts = inputters.collect_feature_vocabs(fields, 'src')
        src_embeddings = build_embeddings(model_opt, src_dict, feature_dicts)
        encoder = build_encoder(model_opt, src_embeddings)
    elif model_opt.model_type == "img":
        encoder = ImageEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout)
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(model_opt.enc_layers,
                               model_opt.brnn,
                               model_opt.rnn_size,
                               model_opt.dropout,
                               model_opt.sample_rate,
                               model_opt.window_size)

    return encoder


def build_generator(model_opt, decoder, vocab):
    # Build Generator.
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)

        # Chris: experimenting with multiple generator layers
        # TODO: multi-layer generator can cause some training tests to fail,
        # TODO: not clear why
        #generator = nn.Sequential(
        #    nn.Linear(model_opt.dec_rnn_size, 256),
        #    nn.LeakyReLU(),
        #    nn.Linear(256, len(vocab)),
        #    gen_func
        #)
        generator = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size, len(vocab)),
            gen_func
        )

        if model_opt.share_decoder_embeddings:
            logger.info('Sharing generator output layer with decoder embeddings')
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        generator = CopyGenerator(model_opt.dec_rnn_size, vocab)

    return generator


def build_decoder_and_generator(model_opt, fields):

    # Build decoder.
    tgt_dict = fields["tgt"].vocab
    feature_dicts = inputters.collect_feature_vocabs(fields, 'tgt')
    tgt_embeddings = build_embeddings(model_opt, tgt_dict,
                                      feature_dicts, for_encoder=False)

    decoder = build_decoder(model_opt, tgt_embeddings)
    generator = build_generator(model_opt, decoder, fields['tgt'].vocab)

    return decoder, generator


def build_attention_bridge(model_opt):
    """
    Create the attention bridge according to user-specified params
    """

    # TODO: expand the options supported by the AttentionBridge
    return AttentionBridge(hidden_size=model_opt.enc_rnn_size,
                           num_attention_heads=model_opt.attention_heads,
                           dec_num_layers=model_opt.dec_layers)


def build_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text", "img", "audio"], \
        "Unsupported model type %s" % model_opt.model_type

    # TODO: loop over fields objects, and taking the "src" from each one
    # for backward compatibility
    if model_opt.rnn_size != -1:
        model_opt.enc_rnn_size = model_opt.rnn_size
        model_opt.dec_rnn_size = model_opt.rnn_size

    # Build embeddings.
    if model_opt.model_type == "text":
        src_fields = [f for n, f in fields['src']]
        assert len(src_fields) == 1
        src_field = src_fields[0]
        src_emb = build_embeddings(model_opt, src_field)
    else:
        src_emb = None

    # Build encoder.
    encoder = build_encoder(model_opt, src_emb)

    # Build decoder.
    tgt_fields = [f for n, f in fields['tgt']]
    assert len(tgt_fields) == 1
    tgt_field = tgt_fields[0]
    tgt_emb = build_embeddings(
        model_opt, tgt_field, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        assert src_field.base_field.vocab == tgt_field.base_field.vocab, \
            "preprocess with -share_vocab if you use share_embeddings"

        tgt_emb.word_lut.weight = src_emb.word_lut.weight

    decoder = build_decoder(model_opt, tgt_emb)

    # Build NMTModel(= encoder + decoder).
    device = torch.device("cuda" if gpu else "cpu")

    # Chris: a different model type for multi-task models
    model = onmt.models.MultiTaskModel(encoder, decoder, model_opt)
    model.model_type = model_opt.model_type

    # Build Generator.
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size,
                      len(fields["tgt"][0][1].base_field.vocab)),
            Cast(torch.float32),
            gen_func
        )
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        assert len(fields["tgt"]) == 1
        tgt_base_field = fields["tgt"][0][1].base_field
        vocab_size = len(tgt_base_field.vocab)
        pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
        generator = CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)

    # Chris: commented while prototyping multi-task
    #if checkpoint is not None:
    #    model.load_state_dict(checkpoint['model'], strict=False)
    #    generator.load_state_dict(checkpoint['generator'], strict=False)
    #else:
    #    if model_opt.param_init != 0.0:
    #        for p in model.parameters():
    #            p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    #        for p in generator.parameters():
    #            p.data.uniform_(-model_opt.param_init, model_opt.param_init)
    #    if model_opt.param_init_glorot:
    #        for p in model.parameters():
    #            if p.dim() > 1:
    #                xavier_uniform_(p)
    #        for p in generator.parameters():
    #            if p.dim() > 1:
    #                xavier_uniform_(p)

    #    if hasattr(model.encoder, 'embeddings'):
    #        model.encoder.embeddings.load_pretrained_vectors(
    #            model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
    #    if hasattr(model.decoder, 'embeddings'):
    #        model.decoder.embeddings.load_pretrained_vectors(
    #            model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)
    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s

        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}
        # end of patch for backward compatibility

        model.load_state_dict(checkpoint['model'], strict=False)
        generator.load_state_dict(checkpoint['generator'], strict=False)
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec)

    model.generator = generator
    model.to(device)
    if model_opt.model_dtype == 'fp16':
        logger.warning('FP16 is experimental, the generated checkpoints may '
                       'be incompatible with a future version')
        model.half()

    return model


def build_multitask_model():
    pass


def build_model(model_opt, opt, fields, checkpoint):
    logger.info('Building model...')
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    logger.info(model)
    return model
