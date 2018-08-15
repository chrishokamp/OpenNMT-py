from __future__ import absolute_import, division, unicode_literals

import sys
import os
import io
import numpy as np
import logging

import csv
import onmt_utils

# Set PATHs
PATH_TO_SENTEVAL = '/home/local/vazquezj/git/SentEval'
#PATH_TO_INFERSENT='/home/local/vazquezj/git/InferSent'
PATH_TO_DATA = '/home/local/vazquezj/git/SentEval/data'
#PATH_TO_VEC = '/home/local/vazquezj/TAITO/ATTATT/some_models/CPU.COCO+caps_acc_72.14_ppl_5.50_e13.pt'
#PATH_TO_VEC2 = '/home/local/vazquezj/Desktop/pruebaATT-ATT/pruebaATT-ATT_acc_9.80_ppl_2144995.48_e2.pt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import torch

#sys.path.insert(0, PATH_TO_INFERSENT)
#import data

def invert_permutation(p):
    p = p.tolist()
    return torch.tensor([p.index(l) for l in range(len(p))])


# SentEval prepare and batcher
def prepare(params, samples):
    '''----------------------------------------------------------
     Onmt patch:
    ----------------------------------------------------------'''
    #params['batch_size']
    #    generate a temporal textfile to pass to Onmt modules
    #import ipdb; ipdb.set_trace(context=5)
    samplefile = "./current-sample.tmp"
    with open(samplefile, "w") as output:
        writer = csv.writer(output, lineterminator='\n', delimiter=" ", quotechar='|')
        writer.writerows(samples)
    #    pass batch textfile -> builds features
    params.data = onmt.io.build_dataset(fields=fields,
                             data_type='text',
                             src_path=samplefile,
                             tgt_path=None,
                             src_dir='',
                             sample_rate='16000',
                             window_size=0.2,
                             window_stride=0.01,
                             window='hamming',
                             use_filter_pred=False)


    #    generate iterator (of size 1) over the dataset
    params.data_iter = onmt.io.OrderedIterator(
            dataset=params.data, device=opt.gpu,
            batch_size=len(params.data), train=False, sort=False,
            sort_within_batch=True, shuffle=False)
    #params.batch_nr = 0

    return


def batcher(params, batch):
    #import ipdb; ipdb.set_trace(context=5)
    batch_size = len(batch)
    #params.batch_nr += 1

    #----------------------------------------------------------
    # Onmt patch: to get embeddings from encoder
    #----------------------------------------------------------
    ''' EXTRACT FEATURES FROM params.data:
    text_features = [ ]
    for sent in batch:
        text_features.append(params.data.extract_text_features(sent))'''
    #onmt.io.make_features(batch,side='src',data_type='text')
    #getattr(data.examples[0], 'src') == data.examples[0].src
    for BATCH in params.data_iter:
        #find the indices of the sentences given in the batch:
        batch_inds = []
        for sentence in batch:
            for example in params.data:
                if (sentence == [i  for i in example.src]):
                    batch_inds.append(example.indices)
                    break
        #import ipdb; ipdb.set_trace()
        # match batch_inds with the BATCH indices (corresponding to original order)
        _, Bindices = torch.sort(BATCH.indices)
        # obtain the sentence vector representation for the batch sentence. In indices[batch_inds]
        batch_data = BATCH.src[0][:,torch.LongTensor(Bindices[batch_inds])]
        batch_data = batch_data.unsqueeze(2)
        batch_data_lengths = BATCH.src[1][torch.LongTensor(Bindices[batch_inds])]
        # sort in decreasing order of the lengths
        batch_data_lengths, length_indices= torch.sort(batch_data_lengths, descending=True)
        batch_data = batch_data[:,length_indices,:]
        # pass the data of the batch through the encoder
        enc_states, memory_bank = model.encoder(batch_data, batch_data_lengths)
        # reorder as the batch was given
        memory_bank = memory_bank[:,invert_permutation(length_indices),:] #shape=[att_heads,batch_size,rnn_size]
    #----------------------------------------------------------
    # make sure embeddings has 1 flattened M matrix per row.
    #import ipdb; ipdb.set_trace()
    memory_bank = memory_bank.transpose(0, 1).contiguous() #shape=[batch_size,att_heads,rnn_size]
    embeddings = [mat.transpose(0,1).flatten().detach() for mat in memory_bank]
    embeddings = np.vstack(embeddings)

    #print('batch number: ' + str(params.batch_nr))
    return embeddings





#################################################
#   STEP1: import the trained model
##################################################
#import ipdb; ipdb.set_trace(context=5)
PATH_TO_ONMT='/home/local/vazquezj/git/ATT-ATT'
sys.path.insert(0, PATH_TO_ONMT)

import argparse
parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
import onmt.opts
onmt.opts.add_md_help_argument(parser)
onmt.opts.translate_opts(parser)
opt = parser.parse_args()

dummy_parser = argparse.ArgumentParser(description='train.py')
onmt.opts.model_opts(dummy_parser)
dummy_opt = dummy_parser.parse_known_args([])[0]

import onmt.ModelConstructor
fields, model, model_opt = \
    onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)
#import ipdb; ipdb.set_trace(context=5)

#encoder = model.encoder
#################################################
#   STEP2: deal with the given batch
#################################################
# #-------------------------
# # example of how a batch is passed to the batcher fn
# #-------------------------
# import csv
# csvfile='/home/local/vazquezj/Desktop/pruebaATT-ATT/batch_example.csv'
#
# with open(csvfile, 'r') as f:
#     reader = csv.reader(f)
#     batch = list(reader)
# #'''The previous batch example was obtained by inserting the following
# #in the SentEval/examples/bow.py in batcher:
# #          with open(csvfile, "w") as output: writer = csv.writer(output, lineterminator='\n'); writer.writerows(batch)
# #'''
# #-------------------------
# batch_size = len(batch)
# import csv
# batchfile = "./current-batch.tmp"
# with open(batchfile, "w") as output:
#     writer = csv.writer(output, lineterminator='\n', delimiter=" ")
#     writer.writerows(batch)
#
#
# data = onmt.io.build_dataset(fields=fields,
#                          data_type='text',
#                          src_path=batchfile,
#                          tgt_path=None,
#                          src_dir='',
#                          sample_rate='16000',
#                          window_size=0.2,
#                          window_stride=0.01,
#                          window='hamming',
#                          use_filter_pred=False)
# #import ipdb; ipdb.set_trace(context=5)
#
# data_iter = onmt.io.OrderedIterator(
#     dataset=data, device=opt.gpu,
#     batch_size=batch_size, train=False, sort=False,
#     sort_within_batch=True, shuffle=False)
#
#
# #################################################
# #   STEP3: run encoder on the batch
# #################################################
# # (1) Run the encoder on the src.
# for BATCH in data_iter:
#     src = onmt.io.make_features(BATCH, side='src', data_type="text")
#     src_lengths = None
#     #if data_type == 'text':
#     #    _, src_lengths = BATCH.src
#     #import ipdb; ipdb.set_trace()
#     _, src_lengths = BATCH.src
#     enc_states, memory_bank = model.encoder(src, src_lengths)
# #return memory_bank
#
# embeddings =  np.vstack(memory_bank.reshape(128,-1).detach())
# #import torch
# #src_lengths = torch.tensor([len(l) for l in batch])
# #enc_states, memory_bank = model.encoder(src, src_lengths)
#



#####################
#   call SentEval
#####################

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    transfer_tasks = ['STS14']
    import ipdb; ipdb.set_trace(context=5)
    results = se.eval(transfer_tasks)
    os.remove(batchfile)
    print(results)
