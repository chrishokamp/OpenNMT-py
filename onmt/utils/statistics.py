""" Statistics calculation utility """
from __future__ import division
import time
import math
import sys
import json

from torch.distributed import get_rank
from onmt.utils.distributed import all_gather_list
from onmt.utils.logging import logger


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by summing values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

        if update_n_src_words:
            self.n_src_words += stat.n_src_words

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def xent(self):
        """ compute word level cross entropy """
        word_level_xent = self.loss / self.n_words
        logger.info(
            'Computing cross-entropy: '
            'dividing loss: {} by num_words: '
            '{}, result: {}'.format(
                self.loss,
                self.n_words,
                word_level_xent))
        return word_level_xent

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def stats(self):
        report = {
            'acc': self.accuracy(),
            'ppl': self.ppl(),
            'xent': self.xent(),
        }
        return report

    def __repr__(self):
        return json.dumps(self.stats())

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        report = dict({
            'step': step,
            'num_steps': num_steps,
            'lr': learning_rate,
            'src_words_per_sec': self.n_src_words / (t + 1e-5),
            'n_words_per_sec': self.n_words / (t + 1e-5),
            'elapsed_time': time.time() - start
        }, **self.stats())

        report_string = \
            (("Step %2d/%5d; acc: %6.2f; ppl: %5.2f; xent: %4.2f; " 
              "lr: %7.5f; Source: %3.0f tok/s Target: %3.0f tok/s; "
              "Time: %6.0f sec")
             % (report['step'], report['num_steps'],
                report['acc'], report['ppl'],
                report['xent'], report['lr'],
                report['src_words_per_sec'], report['n_words_per_sec'],
                report['elapsed_time']))

        logger.info(report_string)
        sys.stdout.flush()

        return report

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
