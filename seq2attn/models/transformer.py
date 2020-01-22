import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
        nhead=8, num_encoder_layers=6, num_decoder_layers=6,
        dim_feedforward=2048, inner_dropout=0.1, inner_activation='relu',
        pos_dropout=0.5, ignore_index=-1,
    ):
        super().__init__()

        self.ignore_index = ignore_index
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, pos_dropout)
        self.inner_transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=inner_dropout,
            activation=inner_activation,
        )
        self.clf = nn.Linear(d_model, tgt_vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.src_emb.weight.data.uniform_(-initrange, initrange)
        self.tgt_emb.weight.data.uniform_(-initrange, initrange)
        self.clf.bias.data.zero_()
        self.clf.weight.data.uniform_(-initrange, initrange)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_variable, input_lengths=None, target_variables=None,
                teacher_forcing_ratio=0):
        src = self.src_emb(input_variable)
        src = self.pos_encoder(src)

        # Remove last token from target, because it is only used in the loss
        tgt = target_variables.get('decoder_output', None)[:, :-1]
        tgt_pad_mask = tgt == self.ignore_index
        tgt = self.tgt_emb(tgt)
        tgt = self.pos_encoder(tgt)

        batch_size, src_max_len, _ = src.shape
        src_pad_mask = torch.arange(src_max_len).expand(batch_size, src_max_len) >= input_lengths.unsqueeze(1)
        tgt_max_len = tgt.shape[1]
        tgt_subsequent_mask = self.inner_transformer.generate_square_subsequent_mask(tgt_max_len)

        output = self.inner_transformer(
            src.transpose(0, 1),
            tgt.transpose(0, 1),
            tgt_mask=tgt_subsequent_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        output = self.clf(output)

        return output, None, None
