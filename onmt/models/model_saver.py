import os
import dill
import torch
import torch.nn as nn

import onmt.inputters

from collections import deque
from onmt.utils.logging import logger


def build_model_saver(model_opt, opt, model, fields, optim):
    model_saver = ModelSaver(opt.save_model,
                             model,
                             model_opt,
                             fields,
                             optim,
                             opt.save_checkpoint_steps,
                             opt.keep_checkpoint)
    return model_saver


class ModelSaverBase(object):
    """
        Base class for model saving operations
        Inherited classes must implement private methods:
            * `_save`
            * `_rm_checkpoint
    """

    def __init__(self, base_path, model, model_opt, fields, optim,
                 save_checkpoint_steps, keep_checkpoint=-1):
        self.base_path = base_path
        self.model = model
        self.model_opt = model_opt
        self.fields = fields
        self.optim = optim
        self.keep_checkpoint = keep_checkpoint
        self.save_checkpoint_steps = save_checkpoint_steps

        if keep_checkpoint > 0:
            self.checkpoint_queue = deque([], maxlen=keep_checkpoint)

    def maybe_save(self, step):
        """
        Main entry point for model saver
        It wraps the `_save` method with checks and apply `keep_checkpoint`
        related logic
        """
        if self.keep_checkpoint == 0:
            return

        if step % self.save_checkpoint_steps != 0:
            return

        chkpt, chkpt_name = self._save(step)

        if self.keep_checkpoint > 0:
            if len(self.checkpoint_queue) == self.checkpoint_queue.maxlen:
                todel = self.checkpoint_queue.popleft()
                self._rm_checkpoint(todel)
            self.checkpoint_queue.append(chkpt_name)

    def _save(self, step):
        """ Save a resumable checkpoint.

        Args:
            step (int): step number

        Returns:
            checkpoint: the saved object
            checkpoint_name: name (or path) of the saved checkpoint
        """
        raise NotImplementedError()

    def _rm_checkpoint(self, name):
        """
        Remove a checkpoint

        Args:
            name(str): name that indentifies the checkpoint
                (it may be a filepath)
        """
        raise NotImplementedError()


class ModelSaver(ModelSaverBase):
    """
        Simple model saver to filesystem
        See:
        https://pytorch.org/docs/stable/notes/serialization.html#recommend-saving-models
        https://pytorch.org/tutorials/beginner/saving_loading_models.html

    """

    def __init__(self, base_path, model, fields, save_checkpoint_steps,
                 model_opt=None, optim=None, keep_checkpoint=0):
        super(ModelSaver, self).__init__(
            base_path, model, model_opt, fields, optim,
            save_checkpoint_steps, keep_checkpoint)

    def create_checkpoint(self):
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)

        #real_generators = (real_model.generators.module
        #                   if isinstance(real_model.generator, nn.DataParallel)
        #                   else real_model.generators)

        #model_state_dict = {k: v for k, v in model_state_dict.items()
        #                    if 'generator' not in k}
        #generator_state_dict = real_generator.state_dict()

        model_state_dict = real_model.state_dict()
        checkpoint = {
            'field_vocabs': self.fields,
            'model_opt': self.model_opt,
            'optim': self.optim,
            'model_state_dict': model_state_dict
        }

        return checkpoint

    def _save(self, step):
        logger.info("Saving checkpoint %s_step_%d.pt" % (self.base_path, step))
        checkpoint_path = '%s_step_%d.pt' % (self.base_path, step)
        checkpoint = self.create_checkpoint()

        torch.save(checkpoint, checkpoint_path, pickle_module=dill)
        return checkpoint, checkpoint_path

    def _rm_checkpoint(self, name):
        os.remove(name)
