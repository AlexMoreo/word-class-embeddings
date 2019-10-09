#taken from https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.helpers import *


class LSTMClassifier(nn.Module):
    # when weights is specified (a tensor of pretrained vectors) it remains fixed and never fine-tuned
    def __init__(self, output_size, hidden_size, vocab_size, learnable_length, pretrained=None, drop_embedding_range=None, drop_embedding_prop=0):
        super(LSTMClassifier, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.drop_embedding_range = drop_embedding_range
        self.drop_embedding_prop = drop_embedding_prop
        assert 0 <= drop_embedding_prop <= 1, 'drop_embedding_prop: wrong range'

        self.pretrained_embeddings, self.learnable_embeddings, self.embedding_length = init_embeddings(pretrained, vocab_size, learnable_length)

        self.lstm = nn.LSTM(self.embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        doc_embedding = self.transform(input)
        logits = self.label(doc_embedding)
        return logits

    def transform(self, input):
        batch_size = input.shape[0]
        input = embed(self, input)
        input = embedding_dropout(input, drop_range=self.drop_embedding_range, p_drop=self.drop_embedding_prop,
                                  training=self.training)
        input = input.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        return final_hidden_state[-1]

    def finetune_pretrained(self):
        self.pretrained_embeddings.requires_grad = True
        self.pretrained_embeddings.weight.requires_grad = True

