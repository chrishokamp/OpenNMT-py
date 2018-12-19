# OpenNMT-py: neural-interlingua branch 
[![Build Status](https://travis-ci.org/OpenNMT/OpenNMT-py.svg?branch=master)](https://travis-ci.org/OpenNMT/OpenNMT-py)  

## About this branch
neural-interlingua is a multilingual implementation of the conjugate attention encoder-decoder NMT ([Cífka and Bojar, 2018](https://arxiv.org/pdf/1805.06536.pdf)), trainable with a language rotating schedule

#### Requirements:
- same as OpenNMT-py 
- for installing in CSC please feel free to use the script `attBridge_setup.sh` (notice that you will need to modify the paths in it)
 
#### USAGE:
##### Training
Same as OpenNMT-py (see [Full documentation](http://opennmt.net/OpenNMT-py/) ) with main differences in the following flags to include multilingual models:
 - `data` (str) [] Give a path to the (preprocessed) data for each language-pair to train, separate them by a space
 - ` src-tgt` (str) [] Indicate the language pairs to train separated by a space (must contain the same number of pairs as the paths given through the `-data` flag). _Example: en-fr de-cs cs-fr_ 

This branch contains the following **NEW** flags for `train.py` routine: 
 - `-use_attention_bridge` () [] Indicates weather to use the compound attention model. Including it will  set to `True` the useage of the self-attention layer
 - `-attention_heads` (int) [default=4] Indicates the number of attention heads (columns) to use on the attention bridge; the `r` parameter from Cífka and Bojar (2018)
 - `-init_decoder` (str) [default = 'rnn_final_state'] Choose a method to initialize decoder. With the final state
                       of the decoder (as in Cifka and Bojar (2018)) or with the average of the heads of the attention bridge (as in [Sennrich et al. (2017)](https://arxiv.org/pdf/1703.04357.pdf) ).
                       choices=['rnn_final_state', 'attention_matrix']
 - `-report_bleu` () [] Using this flag will make the system print the BLEU scores obtained during validation.

##### Translating
To translate using a multilingual model use `translate_multimodel.py` instead of  `translate.py`. This routine is called similarly to `train.py` (see OpenNMT-py documentation), with the following **NEW** flags:
 - `-src_lang` (str) [] source lang of the data to be translated
 - `-tgt_lang` (str) [] target language of the translation
 
### Example:
1- Get some data and **preprocess** it by running the script `data/prep-data.sh`. 
   NOTE: if you ran the `attBridge_setup.sh` script, this has already been done.
   ```
   cd data
   source ./prep-data.sh
   cd ..
   ```
2- **Train** a multilingual model with 10 language pairs (we will use zero-shot translation for `cs <-> en`)
   ```
   python train.py -data data/sample_data/de-cs/data \
                      data/sample_data/fr-cs/data \
                      data/sample_data/de-en/data \
                      data/sample_data/fr-en/data \
                      data/sample_data/cs-de/data \
                      data/sample_data/en-de/data \
                      data/sample_data/fr-de/data \
                      data/sample_data/cs-fr/data \
                      data/sample_data/de-fr/data \
                      data/sample_data/en-fr/data \
                -src_tgt de-cs fr-cs de-en fr-en cs-de en-de fr-de cs-fr de-fr en-fr \
                -save_model ${SAVE_PATH}/MULTILINGUAL          \
                -use_attention_bridge \
                -attention_heads 20 \
                -rnn_size 512 \
                -rnn_type GRU \
                -encoder_type brnn \
                -decoder_type rnn \
                -enc_layers 2 \
                -dec_layers 2 \
                -word_vec_size 512 \
                -global_attention mlp \
                -train_steps 100000 \
                -valid_steps 10000 \
                -optim adam \
                -learning_rate 0.0002 \
                -batch_size 256 \
                -gpuid 0 \
                -save_checkpoint_steps 10000
   ```  

3- **Translate** from src to tgt. You can see that even though we did not train `en-cs` or `cs-en` you can choose this language pair.
   ```
   srun python translate_multimodel.py -model ${SAVE_PATH}/MULTILINGUAL_step_100000.pt \
               -src_lang ${src} \
               -src data/multi30k/dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.${src} \
               -tgt_lang ${tgt} \
               -tgt data/multi30k/dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.${tgt} \
               -report_bleu \
               -gpu 0 \
               -use_attention_bridge \
               -output ${SAVE_PATH}/MULTILINGUAL_prediction_${src}-${tgt}.txt \
               -verbose
   ```
#### Usage Notes:

On the previous example we are using the compound attention as originally proposed, under a encoder-decoder rotational scheme to obtain the language independent sentence representation; i.e., we are simultaneously using the following two

[fig1]: https://github.com/Helsinki-NLP/OpenNMT-py/blob/ATT-ATT/att-att.png "compound attention Figure 1"
[fig2]: https://github.com/Helsinki-NLP/OpenNMT-py/blob/neural-interlingua/data/multi_enc-dec_diagram.png "multiple encoders & decoders Figure 2"



|    Cífka and Bojar's Compound Attention  | Language rotation  |
|:----------------: |:----------------: |
| ![alt text][fig1] | ![alt text][fig2] |
|                   |                   |

 - **ENCODER:** For the encoder, we use 
    1. a 2-layered biGRU; and 
    2. a self-attention layer with `r = attention_heads` attention heads that is shared among all encoders and decoders. 
 
    We adapted the Self-attentive sentence embedding proposed by [Lin et al. (2017)](https://arxiv.org/pdf/1703.03130.pdf) for being use in the OpenNMT environment. Some code is provided in [their GitHub](https://github.com/kaushalshetty/Structured-Self-Attention/blob/master/attention/model.py)

 - **DECODER:** In this case we use a modified version of the OpenNMT `InputFeedRNNDecoder` from `onmt/Models.py`, i.e.,  
    1. a traditional Bahdanau attention layer (1 attention head)
    2. a 2-layered unidirectional GRU. If `-init_decoder` is set to `attention_matrix`,  we set `s_0`, the initial the decoder state, as the average of the columns of the attention bridge - in a similar fashion as [Sennrich et al.(2017)](https://arxiv.org/pdf/1703.04357.pdf). 
 
    i.e., '''
s_0 = tanh(W_init * h_avrg);
'''
    where h_avrg is the average of the sentence embedding, M, (instead of taking the average over the hidden states of the RNN, as in Sennrich(2017))






------------------
# OpenNMT-py: Open-Source Neural Machine Translation
====================================================
[![Build Status](https://travis-ci.org/OpenNMT/OpenNMT-py.svg?branch=master)](https://travis-ci.org/OpenNMT/OpenNMT-py)
[![Run on FH](https://img.shields.io/badge/Run%20on-FloydHub-blue.svg)](https://floydhub.com/run?template=https://github.com/OpenNMT/OpenNMT-py)

## About this branch
Developed on the MT Marathon 2018
Implementation of a conjugate attention encoder-decoder nmt framework ([Cífka and Bojar, 2018](https://arxiv.org/pdf/1805.06536.pdf))
with a language-rotating schedule ([Lu et al., 2018](https://arxiv.org/abs/1804.08198))


This is a [Pytorch](https://github.com/pytorch/pytorch)
port of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
an open-source (MIT) neural machine translation system. It is designed to be research friendly to try out new ideas in translation, summary, image-to-text, morphology, and many other domains. Some companies have proven the code to be production ready.

We love contributions. Please consult the Issues page for any [Contributions Welcome](https://github.com/OpenNMT/OpenNMT-py/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22) tagged post. 

<center style="padding: 40px"><img width="70%" src="http://opennmt.github.io/simple-attn.png" /></center>

Before raising an issue, make sure you read the requirements and the documentation examples.

Unless there is a bug, please use the [Forum](http://forum.opennmt.net) or [Gitter](https://gitter.im/OpenNMT/OpenNMT-py) to ask questions.


Table of Contents
=================
  * [Full Documentation](http://opennmt.net/OpenNMT-py/)
  * [Requirements](#requirements)
  * [Features](#features)
  * [Quickstart](#quickstart)
  * [Run on FloydHub](#run-on-floydhub)
  * [Acknowledgements](#acknowledgements)
  * [Citation](#citation)

## Requirements

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

Note that we currently only support PyTorch 1.0.0

## Features

- [data preprocessing](http://opennmt.net/OpenNMT-py/options/preprocess.html)
- [Inference (translation) with batching and beam search](http://opennmt.net/OpenNMT-py/options/translate.html)
- [Multiple source and target RNN (lstm/gru) types and attention (dotprod/mlp) types](http://opennmt.net/OpenNMT-py/options/train.html#model-encoder-decoder)
- [TensorBoard](http://opennmt.net/OpenNMT-py/options/train.html#logging)
- [Source word features](http://opennmt.net/OpenNMT-py/options/train.html#model-embeddings)
- [Pretrained Embeddings](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-pretrained-embeddings-e-g-glove)
- [Copy and Coverage Attention](http://opennmt.net/OpenNMT-py/options/train.html#model-attention)
- [Image-to-text processing](http://opennmt.net/OpenNMT-py/im2text.html)
- [Speech-to-text processing](http://opennmt.net/OpenNMT-py/speech2text.html)
- ["Attention is all you need"](http://opennmt.net/OpenNMT-py/FAQ.html#how-do-i-use-the-transformer-model)
- [Multi-GPU](http://opennmt.net/OpenNMT-py/FAQ.html##do-you-support-multi-gpu)
- Inference time loss functions.

Beta Features (committed):
- Structured attention
- [Conv2Conv convolution model]
- SRU "RNNs faster than CNN" paper

## Quickstart

[Full Documentation](http://opennmt.net/OpenNMT-py/)


### Step 1: Preprocess the data

```bash
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo
```

We will be working with some example data in `data/` folder.

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are required and used to evaluate the convergence of the training. It usually contains no more than 5000 sentences.


After running the preprocessing, the following files are generated:

* `demo.train.pt`: serialized PyTorch file containing training data
* `demo.valid.pt`: serialized PyTorch file containing validation data
* `demo.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Train the model

```bash
python train.py -data data/demo -save_model demo-model
```

The main train command is quite simple. Minimally it takes a data file
and a save file.  This will run the default model, which consists of a
2-layer LSTM with 500 hidden units on both the encoder/decoder.
If you want to train on GPU, you need to set, as an example:
CUDA_VISIBLE_DEVICES=1,3
`-world_size 2 -gpu_ranks 0 1` to use (say) GPU 1 and 3 on this node only.
To know more about distributed training on single or multi nodes, read the FAQ section.

### Step 3: Translate

```bash
python translate.py -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose
```

Now you have a model which you can use to predict on new data. We do this by running beam search. This will output predictions into `pred.txt`.

!!! note "Note"
    The predictions are going to be quite terrible, as the demo dataset is small. Try running on some larger datasets! For example you can download millions of parallel sentences for [translation](http://www.statmt.org/wmt16/translation-task.html) or [summarization](https://github.com/harvardnlp/sent-summary).

## Alternative: Run on FloydHub

[![Run on FloydHub](https://static.floydhub.com/button/button.svg)](https://floydhub.com/run?template=https://github.com/OpenNMT/OpenNMT-py)

Click this button to open a Workspace on [FloydHub](https://www.floydhub.com/?utm_medium=readme&utm_source=opennmt-py&utm_campaign=jul_2018) for training/testing your code.


## Pretrained embeddings (e.g. GloVe)

Go to tutorial: [How to use GloVe pre-trained embeddings in OpenNMT-py](http://forum.opennmt.net/t/how-to-use-glove-pre-trained-embeddings-in-opennmt-py/1011)

## Pretrained Models

The following pretrained models can be downloaded and used with translate.py.

http://opennmt.net/Models-py/

## Acknowledgements

OpenNMT-py is run as a collaborative open-source project.
The original code was written by [Adam Lerer](http://github.com/adamlerer) (NYC) to reproduce OpenNMT-Lua using Pytorch.

Major contributors are:
[Sasha Rush](https://github.com/srush) (Cambridge, MA)
[Vincent Nguyen](https://github.com/vince62s) (Ubiqus)
[Ben Peters](http://github.com/bpopeters) (Lisbon)
[Sebastian Gehrmann](https://github.com/sebastianGehrmann) (Harvard NLP)
[Yuntian Deng](https://github.com/da03) (Harvard NLP)
[Guillaume Klein](https://github.com/guillaumekln) (Systran)
[Paul Tardy](https://github.com/pltrdy) (Ubiqus / Lium)
[François Hernandez](https://github.com/francoishernandez) (Ubiqus)
[Jianyu Zhan](http://github.com/jianyuzhan) (Shanghai)
and more !

OpentNMT-py belongs to the OpenNMT project along with OpenNMT-Lua and OpenNMT-tf.

## Citation

[OpenNMT: Neural Machine Translation Toolkit](https://arxiv.org/pdf/1805.11462)

[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {Open{NMT}: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```
