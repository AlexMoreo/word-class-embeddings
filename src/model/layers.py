import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class EmbeddingCustom(nn.Module):

    def __init__(self,
                 vocab_size,
                 learnable_length,
                 pretrained=None,
                 drop_embedding_range=None,
                 drop_embedding_prop=0):

        super(EmbeddingCustom, self).__init__()
        assert 0 <= drop_embedding_prop <= 1, 'drop_embedding_prop: wrong range'

        self.vocab_size = vocab_size
        self.drop_embedding_range = drop_embedding_range
        self.drop_embedding_prop = drop_embedding_prop

        pretrained_embeddings = None
        pretrained_length = 0
        if pretrained is not None:
            pretrained_length = pretrained.shape[1]
            assert pretrained.shape[0] == vocab_size, \
                f'pre-trained matrix (shape {pretrained.shape}) does not match with the vocabulary size {vocab_size}'
            pretrained_embeddings = nn.Embedding(vocab_size, pretrained_length)
            # by default, pretrained embeddings are static; this can be modified by calling finetune_pretrained()
            pretrained_embeddings.weight = nn.Parameter(pretrained, requires_grad=False)

        learnable_embeddings = None
        if learnable_length > 0:
            learnable_embeddings = nn.Embedding(vocab_size, learnable_length)

        embedding_length = learnable_length + pretrained_length
        assert embedding_length > 0, '0-size embeddings'

        self.pretrained_embeddings = pretrained_embeddings
        self.learnable_embeddings = learnable_embeddings
        self.embedding_length = embedding_length
        assert self.drop_embedding_range is None or \
               (0<=self.drop_embedding_range[0]<self.drop_embedding_range[1]<=embedding_length), \
            'dropout limits out of range'

    def forward(self, input):
        input = self._embed(input)
        input = self._embedding_dropout(input)
        return input

    def finetune_pretrained(self):
        self.pretrained_embeddings.requires_grad = True
        self.pretrained_embeddings.weight.requires_grad = True

    def _embedding_dropout(self, input):
        if self.droprange():
            # print('\tapplying dropout-range')
            return self._embedding_dropout_range(input)
        elif self.dropfull():
            # print('\tapplying dropout-full')
            return F.dropout(input, p=self.drop_embedding_prop, training=self.training)
        # elif self.dropnone():
        #     print('\tNO dropout')
        return input

    def _embedding_dropout_range(self, input):
        drop_range = self.drop_embedding_range
        p = self.drop_embedding_prop
        if p > 0 and self.training and drop_range is not None:
            drop_from, drop_to = drop_range
            m = drop_to - drop_from     #length of the supervised embedding (or the range)
            l = input.shape[2]          #total embedding length
            corr = (1 - p)
            input[:, :, drop_from:drop_to] = corr * F.dropout(input[:, :, drop_from:drop_to], p=p)
            input /= (1 - (p * m / l))

        return input

    def _embed(self, input):
        input_list = []
        if self.pretrained_embeddings:
            input_list.append(self.pretrained_embeddings(input))
        if self.learnable_embeddings:
            input_list.append(self.learnable_embeddings(input))
        return torch.cat(tensors=input_list, dim=2)

    def dim(self):
        return self.embedding_length

    def dropnone(self):
        if self.drop_embedding_prop == 0:
            return True
        if self.drop_embedding_range is None:
            return True
        return False

    def dropfull(self):
        if self.drop_embedding_prop == 0:
            return False
        if self.drop_embedding_range == [0, self.dim()]:
            return True
        return False

    def droprange(self):
        if self.drop_embedding_prop == 0:
            return False
        if self.drop_embedding_range is None:
            return False
        if self.dropfull():
            return False
        return True


class CNNprojection(nn.Module):

    def __init__(self, embedding_dim, out_channels, kernel_heights=[3, 5, 7], stride=1, padding=0, drop_prob=0.5):
        super(CNNprojection, self).__init__()
        in_channels = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_dim), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_dim), stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_dim), stride, padding)
        self.dropout = nn.Dropout(drop_prob)
        self.out_dimensions = len(kernel_heights) * out_channels

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, input): # input.size() = (batch_size, num_seq, embedding_dim)
        input = input.unsqueeze(1)  # input.size() = (batch_size, 1, num_seq, embedding_length)

        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)

        all_out = torch.cat((max_out1, max_out2, max_out3), 1)  # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)  # fc_in.size()) = (batch_size, num_kernels*out_channels)
        return fc_in

    def dim(self):
        return self.out_dimensions


class LSTMprojection(nn.Module):

    def __init__(self, embedding_dim, hidden_size):
        super(LSTMprojection, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embedding_dim, hidden_size)

    def forward(self, input):  # input.size() = (batch_size, num_seq, embedding_dim)
        batch_size = input.shape[0]
        input = input.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        return final_hidden_state[-1]

    def dim(self):
        return self.hidden_size


class ATTNprojection(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_size):
        super(ATTNprojection, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embedding_dim, hidden_size)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, input):  # input.size() = (batch_size, num_seq, embedding_dim)
        batch_size = input.shape[0]
        input = input.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        output = output.permute(1, 0, 2)  # output.size() = (batch_size, num_seq, hidden_size)
        attn_output = self.attention_net(output, final_hidden_state)
        return attn_output

    def dim(self):
        return self.hidden_size

