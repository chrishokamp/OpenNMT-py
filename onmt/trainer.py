"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

import random

from collections import OrderedDict

import onmt
from copy import deepcopy
import itertools
import torch
import traceback

from onmt.utils.loss import build_loss_from_generator_and_vocab

from onmt.utils.logging import logger

import onmt.utils
from onmt.utils.logging import logger


def build_trainer(opt, model, fields, optim, data_type,
                  generators,
                  tgt_vocabs,
                  device_id=None,
                  model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    # Chris: one loss for every decoder
    train_losses = OrderedDict()
    valid_losses = OrderedDict()
    for tgt_lang, gen in generators.items():
        train_losses[tgt_lang] = \
            build_loss_from_generator_and_vocab(
                gen,
                tgt_vocabs[tgt_lang],
                opt,
                train=True
        )
        valid_losses[tgt_lang] = \
            build_loss_from_generator_and_vocab(
                gen,
                tgt_vocabs[tgt_lang],
                opt,
                train=False
            )

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level
    report_bleu=opt.report_bleu

    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None

    report_manager = onmt.utils.build_report_manager(opt)

    trainer = onmt.Trainer(model, train_losses, valid_losses, optim, trunc_size,
                           shard_size, norm_method,
                           accum_count, accum_steps,
                           n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           model_saver=model_saver if gpu_rank == 0 else None,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opt.model_dtype,
                           earlystopper=earlystopper)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, model, train_losses, valid_losses, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents",
                 accum_count=1,
                 n_gpu=1,
                 gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None, report_bleu=None,
                 use_attention_bridge=False, model_saver=None,
                 average_decay=0, average_every=1,
                 model_dtype='fp32',
                 earlystopper=0):
        # Basic attributes.
        self.model = model
        self.train_losses = train_losses
        self.valid_losses = valid_losses
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = [accum_count]
        # note currently we just have two names for the same attr to keep in sync with upstream
        self.accum_count = accum_count
        self.grad_accum_count = accum_count
        # Note hard-coded accum_steps
        self.accum_steps = [0]
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.report_bleu = report_bleu
        self.use_attention_bridge = use_attention_bridge
        self.last_model = None
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        self.model.train()

    def train(self, train_iter_fcts, valid_iter_fcts, train_steps, valid_steps):
        raise NotImplementedError

    def validate(self, valid_iter, src_tgt, moving_average=None):
        raise NotImplementedError

    def _gradient_accumulation(self, true_batches, normalization, total_stats,
                               report_stats):
        raise NotImplementedError

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None, dataset_name='default_dataset'):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats, dataset_name=dataset_name)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
            self.last_model = self.model_saver.base_path + '_step_' + str(step) +'.pt'

