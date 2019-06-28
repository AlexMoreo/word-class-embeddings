#adapted from https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM_Attn.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from model.helpers import *


class AttentionModel(torch.nn.Module):
    def __init__(self, output_size, hidden_size, vocab_size, learnable_length, pretrained=None, drop_embedding_range=None, drop_embedding_prop=0):
        # when weights is specified (a tensor of pretrained vectors) it remains fixed and never fine-tuned
        super(AttentionModel, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.drop_embedding_range = drop_embedding_range
        self.drop_embedding_prop = drop_embedding_prop
        assert 0 <= drop_embedding_prop <= 1, 'drop_embedding_prop: wrong range'

        self.pretrained_embeddings, self.learnable_embeddings, self.embedding_length = init_embeddings(pretrained,
                                                                                                       vocab_size,
                                                                                                       learnable_length)

        self.lstm = nn.LSTM(self.embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)


    def attention_net(self, lstm_output, final_state):

        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """

        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, input):
        """"
        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)

        """
        attn_output = self.transform(input)
        logits = self.label(attn_output)
        return logits

    def transform(self, input):
        batch_size = input.shape[0]
        input = embed(self, input)
        input = embedding_dropout(input, drop_range=self.drop_embedding_range, p_drop=self.drop_embedding_prop,
                                  training=self.training)
        input = input.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (
        h_0, c_0))  # final_hidden_state.size() = (1, batch_size, hidden_size)
        output = output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)

        attn_output = self.attention_net(output, final_hidden_state)
        return attn_output

    def finetune_pretrained(self):
        self.pretrained_embeddings.requires_grad = True
        self.pretrained_embeddings.weight.requires_grad = True

