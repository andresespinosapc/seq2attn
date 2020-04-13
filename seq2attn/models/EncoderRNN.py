import torch
import torch.nn as nn

from machine.models.baseRNN import BaseRNN


class EncoderRNN(BaseRNN):
    """
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        embedding_size (int): the size of the embedding of input variables
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (default False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
            
    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)

    """

    def __init__(self, vocab_size, max_len, hidden_size, embedding_size,
            input_dropout_p=0, dropout_p=0,
            n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False,
            separate_semantics=False, output_concat='default'):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.output_concat = output_concat
        self.embedding_size = embedding_size
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if separate_semantics:
            self.semantic_embedding = nn.Embedding(vocab_size, embedding_size)
        else:
            self.semantic_embedding = self.embedding
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        embeddings = embedded
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        if self.output_concat == 'russin':
            seq_len, batch_size = output.shape[:2]
            split_dirs_output = output.view(
                seq_len, batch_size,
                self.num_directions, self.hidden_size)
            forward_dir = split_dirs_output[:, :, 0, :]
            backward_dir = split_dirs_output[:, :, 1, :]
            forward_to_concat = torch.cat([forward_dir[-1:], forward_dir[:-1]], dim=0).unsqueeze(2)
            backward_to_concat = torch.cat([backward_dir[1:], backward_dir[:1]], dim=0).unsqueeze(2)
            output = torch.cat([forward_to_concat, backward_to_concat], dim=2).view(output.shape)

        semantic_embeddings = self.semantic_embedding(input_var)
        semantic_embeddings = self.input_dropout(semantic_embeddings)

        return semantic_embeddings, output, hidden
