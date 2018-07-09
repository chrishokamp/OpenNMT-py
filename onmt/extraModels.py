from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq

from onmt.Models import RNNDecoderState

def rnn_factory(rnn_type, **kwargs):
    # Use pytorch version when available.
    no_pack_padded_seq = False
    if rnn_type == "SRU":
        # SRU doesn't support PackedSequence.
        no_pack_padded_seq = True
        rnn = onmt.modules.SRU(**kwargs)
    else:
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn, no_pack_padded_seq

class InnerAttentionEncoder(nn.Module):
    """
    Encoder with inner attention:
        Args:
            rnn_type (str) LSTM or GRU
            bidirectional (bool) whether to use a bidir encoder or not
            num_layers (int)
            hidden_size (int) size of hidden layer
            gpu (list???) gives the gpuid of the device to use
            dropout (double) indicates the dropout proportion choose between [0,1)
            embeddings()
    """
    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size, gpu,
                dropout=0.0, embeddings=None, attentionhops=1, nhid=None, attentionunit=None ):
        super(InnerAttentionEncoder, self).__init__()
        assert embeddings is not None
        #import ipdb; ipdb.set_trace(context=10)
        #self.config = config
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0 # 2 from 2 directions
        self.hidden_dim = hidden_size // 2 # because bidirectional
        self.gpu = gpu
        self.embeddings = embeddings
        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=self.hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        u = hidden_size
        d = hidden_size
        r = attentionhops
        self.ws1 = nn.Linear(d, u, bias=False)
        self.ws2 = nn.Linear(d, r, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        #        self.init_weights()
        self.attention_hops = r

    def forward(self, src, lengths=None, encoder_state=None):
        #import ipdb; ipdb.set_trace(context=10)
        emb = self.embeddings(src)
        s_len, batch_size, emb_dim = emb.size()
        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)
        """
        output, h_n
        output of shape (seq_len, batch, hidden_size * num_directions): tensor containing the output features h_t from the last layer of the GRU, for each t.
        If a torch.nn.utils.rnn.PackedSequence has been given as the input, the output will also be a packed sequence.

        h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t = seq_len
        """
        output, h_n = self.rnn(packed_emb, encoder_state)
        if lengths is not None and not self.no_pack_padded_seq:
            output = unpack(output)[0] #output.size()=[seq_len, batch_size, hidden_size * num_directions]
            #                          #h_n.size()=[num_layers*num_directions, batch_size, hidden_size]
        output2, alphas = self.mixAtt(output)
        #take transpose to match dimensions s.t. r=new_seq_len:
        output3 = torch.transpose(output2, 0, 1).contiguous() #[r,bsz,nhid]
        #import ipdb; ipdb.set_trace(context=10)
        h_avrg = output3.mean(dim=0, keepdim=True)

        return h_avrg, output3
        #return h_n, output3

    def mixAtt(self, outp):
        outp = torch.transpose(outp, 0, 1).contiguous()
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]
        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        raul = alphas.view(-1, size[1])
        alphas = self.softmax(raul)  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas


class InnerAttentionDecoder(nn.Module):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="mlp",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 attentionhops=1, reuse_copy_attn=False):
        super(InnerAttentionDecoder, self).__init__()
        #import ipdb; ipdb.set_trace(context=5)
        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_dim = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        '''Decoder initialization: '''
        u = self.hidden_dim
        r = attentionhops
        self.attention_hops = r
        self.W_init = nn.Linear(u,num_layers*u , bias=False)
        self.tanh = nn.Tanh()
        # Build the RNN.
        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=self.hidden_dim,
                        num_layers=(num_layers ),# // 2),
                        dropout=dropout)
        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            self.hidden_dim, coverage=coverage_attn,
            attn_type=attn_type
        )
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
        if copy_attn:
            self._copy = True

        self._reuse_copy_attn = reuse_copy_attn


    def forward(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.Models.DecoderState`):
                 decoder state object to initialize the decoder
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        #import ipdb; ipdb.set_trace()
        # Check
        assert isinstance(state, RNNDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        _, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)
        # END

        # Run the forward pass of the RNN.
        decoder_final, decoder_outputs, attns = self._run_forward_pass(
            tgt, memory_bank, state, memory_lengths=memory_lengths)

        # Update the state with the result.
        final_output = decoder_outputs[-1]
        coverage = None
        if "coverage" in attns:
            coverage = attns["coverage"][-1].unsqueeze(0)
        state.update_state(decoder_final, final_output.unsqueeze(0), coverage)

        # Concatenates sequence of tensors along a new dimension.
        decoder_outputs = torch.stack(decoder_outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return decoder_outputs, state, attns

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                 [len x batch x nfeats].
            memory_bank (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            decoder_final (Variable): final hidden state from the decoder.
            decoder_outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """

        #import ipdb; ipdb.set_trace(context=5)
        # Initialize local and return variables.
        attns = {}
        emb = self.embeddings(tgt)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, decoder_final = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, decoder_final = self.rnn(emb, state.hidden)

        # Check
        tgt_len, tgt_batch, _ = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)
        # END


        # Calculate the attention.
        M_size = torch.zeros(memory_lengths.size()) + self.attention_hops
        decoder_outputs, p_attn = self.attn(
            rnn_output.transpose(0, 1).contiguous(),
            memory_bank.transpose(0, 1),
            memory_lengths=M_size
        )
        attns["std"] = p_attn

        # Calculate the context gate.
        if self.context_gate is not None:
            decoder_outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                decoder_outputs.view(-1, decoder_outputs.size(2))
            )
            decoder_outputs = \
                decoder_outputs.view(tgt_len, tgt_batch, self.hidden_size)

        decoder_outputs = self.dropout(decoder_outputs)
        return decoder_final, decoder_outputs, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    def init_decoder_state(self, src, memory_bank, encoder_final):
        '''
        initialize the decoder state, `s_0`.
        We use a similar s_0 as in rSennrich et al.(2017) "Nematus":
             - s_0 = tanh(W_init * h_avrg);
                  where h_avrg is the average of the sentence embedding, M,
                  (instead of taking the average over the hidden states of
                   the RNN, as in rSennrich(2017) over the )
        '''
        import ipdb; ipdb.set_trace()
        s_0 = self.tanh(self.W_init(encoder_final))
        ss = s_0[:,:,:self.hidden_dim]
        for i in range(self.num_layers - 1):
            init = (i+1) * self.hidden_dim
            end = (i+2) * self.hidden_dim
            print(init,end)
            ss = torch.cat([ss,s_0[:,:,init:end]])
        s_0
        return RNNDecoderState(self.hidden_dim, s_0)


######################################################
######################################################



class BiLSTMInnerAttentionEncoder(nn.Module):
    """
    Bidirectional LSTM with inner attention
                 self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False

    """
    def __init__(self, config):
        super(BiLSTMInnerAttentionEncoder, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.gpu = config.gpu
        self.rnn1 = nn.GRU(input_size=config.embed_dim,
                           hidden_size=self.hidden_dim,
                           num_layers=config.layers,
                           dropout=config.dropout,
                           bidirectional=True)

        self.key_projection = nn.Linear(2*self.hidden_dim,
                                   2*self.hidden_dim,
                                   bias=False)
        self.proj_query = nn.Linear(2*self.hidden_dim,
                                    2*self.hidden_dim,
                                    bias=False)
        self.projection = nn.Linear(2*self.hidden_dim,
                                    2*self.hidden_dim,
                                    bias=False)
        self.query = nn.Embedding(2, 2*self.hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        h_0 = c_0 = Variable(inputs.data.new(self.config.cells,
                                             batch_size,
                                             self.hidden_dim).zero_())
        hidden_states, (last_h, last_c) = self.rnn1(inputs, (h_0, c_0))

        emb = self.attention(hidden_states, batch_size, temp=2)

        return emb


    def attention(self, hidden_states, batch_size, temp=2):
        output = hidden_states.transpose(0,1).contiguous()
        output_proj = self.projection(output.view(-1, 2*self.hidden_dim)).view(batch_size, -1, 2*self.hidden_dim)
        key = self.key_projection(output.view(-1, 2*self.hidden_dim)).view(batch_size, -1, 2*self.hidden_dim)
        key = torch.tanh(key)
        out = self.query(Variable(torch.LongTensor(batch_size*[0]).cuda(device=self.gpu))).unsqueeze(2)
        keys = key.bmm(out).squeeze(2) / temp
        keys = keys + ((keys == 0).float()*-1000)
        alphas = self.softmax(keys).unsqueeze(2).expand_as(key)
        atn_embed = torch.sum(alphas * output_proj, 1).squeeze(1)

        return atn_embed



######################################################
######## ATTEMPT_2:  L I N  et  al.'s    C O D E #####
######################################################

class BiLSTM(nn.Module):

    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        self.encoder = nn.Embedding(config['ntoken'], config['ninp'])
        self.bilstm = nn.LSTM(config['ninp'], config['nhid'], config['nlayers'], dropout=config['dropout'],
                              bidirectional=True)
        self.nlayers = config['nlayers']
        self.nhid = config['nhid']
        self.pooling = config['pooling']
        self.dictionary = config['dictionary']
#        self.init_weights()
        self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0
        if os.path.exists(config['word-vector']):
            print('Loading word vectors from', config['word-vector'])
            vectors = torch.load(config['word-vector'])
            assert vectors[2] >= config['ninp']
            vocab = vectors[0]
            vectors = vectors[1]
            loaded_cnt = 0
            for word in self.dictionary.word2idx:
                if word not in vocab:
                    continue
                real_id = self.dictionary.word2idx[word]
                loaded_id = vocab[word]
                self.encoder.weight.data[real_id] = vectors[loaded_id][:config['ninp']]
                loaded_cnt += 1
            print('%d words from external word vectors loaded.' % loaded_cnt)

    # note: init_range constraints the value of initial weights
    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        emb = self.drop(self.encoder(inp))
        outp = self.bilstm(emb, hidden)[0]
        if self.pooling == 'mean':
            outp = torch.mean(outp, 0).squeeze()
        elif self.pooling == 'max':
            outp = torch.max(outp, 0)[0].squeeze()
        elif self.pooling == 'all' or self.pooling == 'all-word':
            outp = torch.transpose(outp, 0, 1).contiguous()
        return outp, emb

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()))


class SelfAttentiveEncoder(nn.Module):

    def __init__(self, config):
        super(SelfAttentiveEncoder, self).__init__()
        self.bilstm = BiLSTM(config)
        self.drop = nn.Dropout(config['dropout'])
        self.ws1 = nn.Linear(config['nhid'] * 2, config['attention-unit'], bias=False)
        self.ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dictionary = config['dictionary']
#        self.init_weights()
        self.attention_hops = config['attention-hops']

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        outp = self.bilstm.forward(inp, hidden)[0]
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]
        transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
        transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        penalized_alphas = alphas + (
            -10000 * (concatenated_inp == self.dictionary.word2idx['<pad>']).float())
            # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

    def init_hidden(self, bsz):
        return self.bilstm.init_hidden(bsz)



######################################################
######## ATTEMPT_1:  A A R N E 's    C O D E #########
######################################################

'''
class InnerAttentionEncoder(nn.Module):
    """
    Encoder with inner attention:
        Args:
            rnn_type (str) LSTM or GRU
            bidirectional (bool) whether to use a bidir encoder or not
            num_layers (int)
            hidden_size (int) size of hidden layer
            gpu (list???) gives the gpuid of the device to use
            dropout (double) indicates the dropout proportion choose between [0,1)
            embeddings()
    """
    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size, gpu, dropout=0.0, embeddings=None):
        super(InnerAttentionEncoder, self).__init__()
        assert embeddings is not None

        #import ipdb; ipdb.set_trace(context=10)
        #self.config = config
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0 # 2 from 2 directions
        self.hidden_dim = hidden_size // 2 # because bidirectional
        self.gpu = gpu
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=self.hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        #if rnn_type == 'GRU':
        #    self.rnn = nn.GRU(input_size=embeddings.embedding_size,
        #                   hidden_size=self.hidden_dim,
        #                   num_layers=num_layers,
        #                   dropout=dropout,
        #                   bidirectional=bidirectional)
        #else:
        #    self.rnn = nn.LSTM(input_size=config.embed_dim,
        #                   hidden_size=self.hidden_dim,
        #                   num_layers=config.layers,
        #                   dropout=config.dropout,
        #                   bidirectional=True)


        self.key_projection = nn.Linear(2*self.hidden_dim,
                                   2*self.hidden_dim,
                                   bias=False)
        self.proj_query = nn.Linear(2*self.hidden_dim,
                                    2*self.hidden_dim,
                                    bias=False)
        self.projection = nn.Linear(2*self.hidden_dim,
                                    2*self.hidden_dim,
                                    bias=False)
        self.query = nn.Embedding(2, 2*self.hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, lengths=None, encoder_state=None):
        import ipdb; ipdb.set_trace(context=10)

        emb = self.embeddings(src)
        s_len, batch_size, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        hidden_states, encoder_final = self.rnn(packed_emb, encoder_state)
        if lengths is not None and not self.no_pack_padded_seq:
            hidden_states = unpack(hidden_states)[0]

        emb, alphas = self.attention(hidden_states, batch_size, temp=2)
        return emb, alphas

    def attention(self, hidden_states, batch_size, temp=2):
        import ipdb; ipdb.set_trace()
        output = hidden_states.transpose(0,1).contiguous()
        output_proj = self.projection(output.view(-1, 2*self.hidden_dim)).view(batch_size, -1, 2*self.hidden_dim)
        key = self.key_projection(output.view(-1, 2*self.hidden_dim)).view(batch_size, -1, 2*self.hidden_dim)
        key = torch.tanh(key)
        #out = self.query(Variable(torch.LongTensor(batch_size*[0]).cuda(device=self.gpu))).unsqueeze(2)
        out = self.query(Variable(torch.LongTensor(batch_size*[0]))).unsqueeze(2)
        keys = key.bmm(out).squeeze(2) / temp
        keys = keys + ((keys == 0).float()*-1000)
        alphas = self.softmax(keys).unsqueeze(2).expand_as(key)
        atn_embed = torch.sum(alphas * output_proj, 1).squeeze(1)

        return atn_embed, alphas

'''
