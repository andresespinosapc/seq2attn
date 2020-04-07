"""Train Seq2Attn model."""

from comet_ml import Experiment, OfflineExperiment
from callbacks import CometLogger

import argparse
from collections import OrderedDict
import logging
import os

import torch
import torchtext

from seq2attn.models import EncoderRNN
from seq2attn.models import Seq2AttnDecoder
from seq2attn.models import Seq2seq
from seq2attn.models import Transformer

from machine.dataset import SourceField, get_standard_iter
from machine.dataset import TargetField
from machine.loss import NLLLoss
from machine.metrics import FinalTargetAccuracy
from machine.metrics import SequenceAccuracy
from machine.metrics import SymbolRewritingAccuracy
from machine.metrics import WordAccuracy
from machine.trainer import SupervisedTrainer
from machine.util.checkpoint import Checkpoint

import random
import numpy as np
# from machine.tasks import get_task
import sys
sys.path.insert(0, os.path.abspath('../machine-tasks'))
from tasks import get_task
from loss import L1Loss

comet_args = {
    'project_name': 'attentive-guidance',
    'workspace': 'andresespinosapc',
}
if os.environ.get('COMET_DISABLE'):
    comet_args['disabled'] = True
    comet_args['api_key'] = ''
if os.environ.get('COMET_OFFLINE'):
    comet_args['api_key'] = ''
    comet_args['offline_directory'] = 'comet_offline'
    experiment = OfflineExperiment(**comet_args)
else:
    experiment = Experiment(**comet_args)

def log_comet_parameters(opt):
    opt_dict = vars(opt)
    for key in opt_dict.keys():
        experiment.log_parameter(key, opt_dict[key])

TASK_DEFAULT_PARAMS = {
    'task_defaults': {
        'batch_size': 128,
        'k': 3,
        'max_len': 60,
        'patience': 5,
        'epochs': 20,
    },
    'simple': {
        'batch_size': 1,
        'embedding_size': 256,
        'hidden_size': 256,
        'n_layers': 1,
        'rnn_cell': 'gru',
        'attention': 'pre-rnn',
        'max_len': 50,
        'dropout_p_encoder': 0.5,
        'dropout_p_decoder': 0.5,
    },
    'Hupkes_2019_lookup_baseline': {
        'full_attention_focus': False,
        'sample_train': 'softmax',
        'attn_vals': 'outputs',
        'output_value': 'context',
        'batch_size': 1,
        'embedding_size': 128,
        'hidden_size': 512,
        'n_layers': 1,
        'rnn_cell': 'gru',
        'attention': 'pre-rnn',
        'attention_method': 'mlp',
        'max_len': 50,
        'dropout_p_encoder': 0.5,
        'dropout_p_decoder': 0.5,
    },
    'Hupkes_2019_lookup_seq2attn': {
        'full_attention_focus': True,
        'sample_train': 'gumbel_st',
        'attn_vals': 'embeddings',
        'output_value': 'decoder_output',
        'batch_size': 1,
        'embedding_size': 256,
        'hidden_size': 256,
        'n_layers': 1,
        'rnn_cell': 'gru',
        'attention': 'pre-rnn',
        'attention_method': 'mlp',
        'max_len': 50,
        'dropout_p_encoder': 0.5,
        'dropout_p_decoder': 0.5,
    },
    'Hupkes_2019_SCAN_baseline': {
        'full_attention_focus': False,
        'sample_train': 'softmax',
        'attn_vals': 'outputs',
        'output_value': 'context',
        'batch_size': 1,
        'embedding_size': 200,
        'hidden_size': 200,
        'n_layers': 2,
        'rnn_cell': 'lstm',
        'attention': 'pre-rnn',
        'attention_method': 'mlp',
        'max_len': 50,
        'dropout_p_encoder': 0.5,
        'dropout_p_decoder': 0.5,
    },
    'Hupkes_2019_SCAN_seq2attn': {
        'full_attention_focus': True,
        'sample_train': 'gumbel_st',
        'attn_vals': 'embeddings',
        'output_value': 'decoder_output',
        'batch_size': 1,
        'embedding_size': 512,
        'hidden_size': 512,
        'n_layers': 1,
        'rnn_cell': 'gru',
        'attention': 'pre-rnn',
        'attention_method': 'mlp',
        'max_len': 50,
        'dropout_p_encoder': 0.5,
        'dropout_p_decoder': 0.5,
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--task', type=str, choices=[
    'lookup',
    'long_lookup',
    'long_lookup_oneshot',
    'long_lookup_reverse',
    'noisy_long_lookup_multi',
    'noisy_long_lookup_single',
    'long_lookup_intermediate_noise',
    'symbol_rewriting',
    'SCAN_simple',
    'SCAN_length',
    'SCAN_addprim_turn_left',
    'SCAN_addprim_jump',
    ])
parser.add_argument('--default_params_key', type=str, choices=list(TASK_DEFAULT_PARAMS.keys()), default='task_defaults')
parser.add_argument('--l1_loss_inputs', type=str, nargs='*',
    choices=['encoder_hidden', 'model_parameters'], default=[])
parser.add_argument('--scale_l1_loss', type=float, default=1.)
parser.add_argument('--auto_print_every', action='store_true')
parser.add_argument('--model', type=str, default='seq2attn', choices=['seq2attn', 'transformer'])

parser.add_argument('--train', help='Training data')
parser.add_argument('--dev', help='Development data')
parser.add_argument('--monitor', nargs='+', default=[], help='Data to monitor during training')
parser.add_argument('--output_dir', default='../models', help='Path to model directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--epochs', type=int, help='Number of epochs', default=6)
parser.add_argument('--optim', type=str, default='adam', help='Choose optimizer', choices=['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'sgd'])
parser.add_argument('--max_len', type=int, help='Maximum sequence length')
parser.add_argument('--lower', action='store_true', help='Whether to lowercase the text in this field')
parser.add_argument('--rnn_cell', type=str, help="Chose type of rnn cell")
parser.add_argument('--bidirectional', action='store_true', help="Flag for bidirectional encoder")
parser.add_argument('--embedding_size', type=int, help='Embedding size')
parser.add_argument('--hidden_size', type=int, help='Hidden layer size')
parser.add_argument('--n_layers', type=int, help='Number of RNN layers in both encoder and decoder')
parser.add_argument('--src_vocab', type=int, help='source vocabulary size', default=50000)
parser.add_argument('--tgt_vocab', type=int, help='target vocabulary size', default=50000)
parser.add_argument('--dropout_p_encoder', type=float, help='Dropout probability for the encoder')
parser.add_argument('--dropout_p_decoder', type=float, help='Dropout probability for the decoder')
parser.add_argument('--teacher_forcing_ratio', type=float, help='Teacher forcing ratio', default=0.2)
parser.add_argument('--attention', choices=['pre-rnn'])
parser.add_argument('--attention_method', choices=['dot', 'mlp', 'concat'], default=None)
parser.add_argument('--metrics', nargs='+', default=['seq_acc'], choices=['word_acc', 'seq_acc', 'target_acc', 'sym_rwr_acc'], help='Metrics to use')
parser.add_argument('--batch_size', type=int, help='Batch size')
parser.add_argument('--eval_batch_size', type=int, help='Batch size', default=128)
parser.add_argument('--lr', type=float, help='Learning rate, recommended settings.\nrecommended settings: adam=0.001 adadelta=1.0 adamax=0.002 rmsprop=0.01 sgd=0.1', default=0.001)
parser.add_argument('--ignore_output_eos', action='store_true', help='Ignore end of sequence token during training and evaluation')

parser.add_argument('--load_checkpoint', help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--save_every', type=int, help='Every how many batches the model should be saved', default=100)
parser.add_argument('--print_every', type=int, help='Every how many batches to print results', default=100)
parser.add_argument('--resume', action='store_true', help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', default='info', help='Logging level.')
parser.add_argument('--write-logs', help='Specify file to write logs to after training')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')

# Arguments for the Seq2Attn model
parser.add_argument('--sample_train', type=str, choices=['softmax', 'softmax_st', 'gumbel', 'gumbel_st', 'sparsemax'], help='During training, activate the attention vector using Softmax (ST), Gumbel-Softmax (ST) or Sparsemax')
parser.add_argument('--sample_infer', type=str, default='argmax', choices=['softmax', 'softmax_st', 'gumbel', 'gumbel_st', 'sparsemax', 'argmax'], help='During testing, activate the attention vector using Softmax (ST), Gumbel-Softmax (ST), argmax or Sparsemax')
parser.add_argument('--initial_temperature', type=float, default=1., help='(Initial) temperature to use for Gumbel-Softmax or Softmax ST')
parser.add_argument('--learn_temperature', type=str, default='no', choices=['no', 'latent', 'conditioned'], help='Whether the temperature should be a learnable parameter. And whether it should be conditioned')
parser.add_argument('--attn_vals', type=str, choices=['outputs', 'embeddings'], help="Attend to hidden states or embeddings.")
parser.add_argument('--full_attention_focus', type=bool, help='Indicate whether to multiply the hidden state of the decoder with the context vector')
parser.add_argument('--output_value', type=str, default=None, choices=['decoder_output', 'context'], help='Which is the output vector of the decoder')
parser.add_argument('--transcoder_hidden_activation', type=str, default=None, choices=['none', 'gumbel_st'], help='Apply an activation to the transcoder hidden state in every step')
parser.add_argument('--tha_initial_temperature', type=float, default=1., help='(Initial) temperature to use for Gumbel-Softmax or Softmax ST')
parser.add_argument('--decoder_hidden_activation', type=str, default=None, choices=['none', 'gumbel_st'], help='Apply an activation to the decoder hidden state in every step')
parser.add_argument('--dha_initial_temperature', type=float, default=1., help='(Initial) temperature to use for Gumbel-Softmax or Softmax ST')
parser.add_argument('--decoder_hidden_override', type=str, default=None, choices=['zeros', 'context'], help='Override decoder hidden state with another vector')

parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for CometML logging')

opt = parser.parse_args()

if opt.task is not None:
    task = get_task(opt.task)
    opt.train = task.train_path
    opt.dev = task.valid_path
    opt.monitor = task.test_paths

for key, value in TASK_DEFAULT_PARAMS[opt.default_params_key].items():
    if key not in opt.__dict__:
        raise ValueError('Param %s not in argparse options' % (key))
    if getattr(opt, key) is None:
        setattr(opt, key, value)
    else:
        logging.info('Overriding default parameter %s value %s with %s' % (
            key, value, getattr(opt, key),
        ))

log_comet_parameters(opt)

IGNORE_INDEX = -1
use_output_eos = not opt.ignore_output_eos

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.resume and not opt.load_checkpoint:
    parser.error('load_checkpoint argument is required to resume training from checkpoint')

if not opt.attention and opt.attention_method:
    parser.error("Attention method provided, but attention is not turned on")

if opt.attention and not opt.attention_method:
    parser.error("Attention turned on, but no attention method provided")

if opt.exp_name is None and not comet_args.get('disabled'):
    parser.error('Please provide exp_name if logging to CometML')
experiment.set_name(opt.exp_name)

if torch.cuda.is_available():
        logging.info("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

if opt.attention:
    if not opt.attention_method:
        logging.info("No attention method provided. Using DOT method.")
        opt.attention_method = 'dot'

# Set random seed
if opt.random_seed:
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.random_seed)

############################################################################
# Prepare dataset
src = SourceField(lower=opt.lower)
tgt = TargetField(include_eos=use_output_eos, lower=opt.lower)

tabular_data_fields = [('src', src), ('tgt', tgt)]

max_len = opt.max_len

def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

# generate training and testing data
train = torchtext.data.TabularDataset(
    path=opt.train, format='tsv',
    fields=tabular_data_fields,
    filter_pred=len_filter
)
train = get_standard_iter(train, batch_size=opt.batch_size)

if opt.auto_print_every:
    opt.print_every = int(len(train) / 2)

if opt.dev:
    dev = torchtext.data.TabularDataset(
        path=opt.dev, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    )
    dev = get_standard_iter(dev, batch_size=opt.eval_batch_size)

else:
    dev = None

monitor_data = OrderedDict()
for dataset in opt.monitor:
    m = torchtext.data.TabularDataset(
        path=dataset, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter)
    m = get_standard_iter(m, batch_size=opt.eval_batch_size)
    monitor_data[dataset] = m

#################################################################################
# prepare model

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.output_dir, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.output_dir, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model

    input_vocab = checkpoint.input_vocab
    src.vocab = input_vocab

    output_vocab = checkpoint.output_vocab
    tgt.vocab = output_vocab
    tgt.eos_id = tgt.vocab.stoi[tgt.SYM_EOS]
    tgt.sos_id = tgt.vocab.stoi[tgt.SYM_SOS]

else:
    # build vocabulary
    src.build_vocab(train.dataset, max_size=opt.src_vocab)
    tgt.build_vocab(train.dataset, max_size=opt.tgt_vocab)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # Initialize model
    if opt.model == 'seq2attn':
        hidden_size = opt.hidden_size
        decoder_hidden_size = hidden_size*2 if opt.bidirectional else hidden_size
        seq2attn_encoder = EncoderRNN(len(src.vocab),
                                    max_len,
                                    hidden_size,
                                    opt.embedding_size,
                                    dropout_p=opt.dropout_p_encoder,
                                    n_layers=opt.n_layers,
                                    bidirectional=opt.bidirectional,
                                    rnn_cell=opt.rnn_cell,
                                    variable_lengths=True)
        decoder = Seq2AttnDecoder(
                            len(tgt.vocab), max_len, decoder_hidden_size,
                            dropout_p=opt.dropout_p_decoder,
                            n_layers=opt.n_layers,
                            use_attention=opt.attention,
                            attention_method=opt.attention_method,
                            bidirectional=opt.bidirectional,
                            rnn_cell=opt.rnn_cell,
                            eos_id=tgt.eos_id,
                            sos_id=tgt.sos_id,
                            embedding_dim=opt.embedding_size,
                            sample_train=opt.sample_train,
                            sample_infer=opt.sample_infer,
                            initial_temperature=opt.initial_temperature,
                            learn_temperature=opt.learn_temperature,
                            attn_vals=opt.attn_vals,
                            full_attention_focus=opt.full_attention_focus,
                            output_value=opt.output_value,
                            transcoder_hidden_activation=opt.transcoder_hidden_activation,
                            tha_initial_temperature=opt.tha_initial_temperature,
                            decoder_hidden_activation=opt.decoder_hidden_activation,
                            dha_initial_temperature=opt.dha_initial_temperature,
                            decoder_hidden_override=opt.decoder_hidden_override)
        seq2seq = Seq2seq(seq2attn_encoder, decoder)
    elif opt.model == 'transformer':
        seq2seq = Transformer(
            src_vocab_size=len(src.vocab),
            tgt_vocab_size=len(tgt.vocab),
            ignore_index=output_vocab.stoi[tgt.pad_token],
        )
    seq2seq.to(device)

    for param in seq2seq.named_parameters():
        name, data = param[0], param[1].data
        # Don't reinitialize temperature
        if 'temperature' not in name:
            data.uniform_(-0.08, 0.08)

input_vocabulary = input_vocab.itos
output_vocabulary = output_vocab.itos

##############################################################################
# train model

# Prepare loss and metrics
pad = output_vocab.stoi[tgt.pad_token]
losses = [NLLLoss(ignore_index=pad)]
loss_weights = [1.]

for l1_loss_input in opt.l1_loss_inputs:
    losses.append(L1Loss(input_name=l1_loss_input))
    loss_weights.append(opt.scale_l1_loss)

for loss in losses:
    loss.to(device)

metrics = []
if 'word_acc' in opt.metrics:
    metrics.append(WordAccuracy(ignore_index=pad))
if 'seq_acc' in opt.metrics:
    metrics.append(SequenceAccuracy(ignore_index=pad))
if 'target_acc' in opt.metrics:
    metrics.append(FinalTargetAccuracy(ignore_index=pad, eos_id=tgt.eos_id))
if 'sym_rwr_acc' in opt.metrics:
    metrics.append(SymbolRewritingAccuracy(input_vocab=input_vocab,
                                           output_vocab=output_vocab,
                                           use_output_eos=use_output_eos,
                                           output_sos_symbol=tgt.SYM_SOS,
                                           output_pad_symbol=tgt.pad_token,
                                           output_eos_symbol=tgt.SYM_EOS,
                                           output_unk_symbol=tgt.unk_token))

checkpoint_path = os.path.join(opt.output_dir, opt.load_checkpoint) if opt.resume else None

# create trainer
expt_dir = os.path.join(opt.output_dir, experiment.get_key())
t = SupervisedTrainer(expt_dir=expt_dir)

seq2seq, logs = t.train(model=seq2seq,
                        data=train,
                        dev_data=dev,
                        monitor_data=monitor_data,
                        num_epochs=opt.epochs,
                        optimizer=opt.optim,
                        teacher_forcing_ratio=opt.teacher_forcing_ratio,
                        learning_rate=opt.lr,
                        resume_training=opt.resume,
                        checkpoint_path=checkpoint_path,
                        losses=losses,
                        metrics=metrics,
                        loss_weights=loss_weights,
                        checkpoint_every=opt.save_every,
                        print_every=opt.print_every,
                        random_seed=opt.random_seed,
                        custom_callbacks=[CometLogger(experiment)])

if opt.write_logs:
    output_path = os.path.join(opt.output_dir, opt.write_logs)
    logs.write_to_file(output_path)
