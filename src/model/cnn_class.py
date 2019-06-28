#adapted from https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/CNN.py
import torch.nn as nn
from torch.nn import functional as F
from model.helpers import *


class CNN(nn.Module):
    """
    The idea of the Convolutional Neural Netwok for Text Classification is very simple. We perform convolution operation on the embedding matrix
    whose shape for each batch is (num_seq, embedding_length) with kernel of varying height but constant width which is same as the embedding_length.
    We will be using ReLU activation after the convolution operation and then for each kernel height, we will use max_pool operation on each tensor
    and will filter all the maximum activation for every channel and then we will concatenate the resulting tensors. This output is then fully connected
    to the output layers consisting two units which basically gives us the logits for both positive and negative classes.
    """

    def __init__(self, output_size, out_channels, vocab_size, learnable_length, pretrained=None, kernel_heights=[3,5,7],
                 stride=1, padding=0, drop_prob=0.5,
                 # word_drop=0.1, unk_id=0, pad_id=1,
                 drop_embedding_range=None, drop_embedding_prop=0):

        super(CNN, self).__init__()
        self.output_size = output_size
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.stride = stride
        self.padding = padding
        self.vocab_size = vocab_size
        # self.unk_id = unk_id
        # self.pad_id = pad_id
        self.drop_embedding_range = drop_embedding_range
        self.drop_embedding_prop = drop_embedding_prop
        assert 0<=drop_embedding_prop<=1, 'drop_embedding_prop: wrong range'

        self.pretrained_embeddings, self.learnable_embeddings, self.embedding_length = init_embeddings(pretrained, vocab_size, learnable_length)

        in_channels=1
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], self.embedding_length), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], self.embedding_length), stride, padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], self.embedding_length), stride, padding)
        self.dropout = nn.Dropout(drop_prob)
        # self.word_drop = word_drop
        self.label = nn.Linear(len(kernel_heights) * out_channels, output_size)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)

        return max_out

    def forward(self, input):
        doc_embedding = self.transform(input)
        logits = self.label(doc_embedding)
        return logits

    def transform(self, input):
        # if self.training:
        #     input = word_dropout(input, self.pad_id, self.unk_id, p_drop=self.word_drop)

        input = embed(self, input)
        input = embedding_dropout(input, drop_range=self.drop_embedding_range, p_drop=self.drop_embedding_prop, training=self.training)
        input = input.unsqueeze(1) # input.size() = (batch_size, 1, num_seq, embedding_length)

        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)

        all_out = torch.cat((max_out1, max_out2, max_out3), 1) # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out) # fc_in.size()) = (batch_size, num_kernels*out_channels)
        return fc_in

    def finetune_pretrained(self):
        self.pretrained_embeddings.requires_grad = True
        self.pretrained_embeddings.weight.requires_grad = True
