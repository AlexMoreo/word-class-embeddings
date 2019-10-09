import torch
import torch.nn as nn
from torch.nn import functional as F


def init_embeddings(pretrained, vocab_size, learnable_length):
    pretrained_embeddings = None
    pretrained_length = 0
    if pretrained is not None:
        pretrained_length = pretrained.shape[1]
        assert pretrained.shape[0] == vocab_size, 'pre-trained matrix does not match with the vocabulary size'
        pretrained_embeddings = nn.Embedding(vocab_size, pretrained_length)
        pretrained_embeddings.weight = nn.Parameter(pretrained, requires_grad=False)

    learnable_embeddings = None
    if learnable_length > 0:
        learnable_embeddings = nn.Embedding(vocab_size, learnable_length)

    embedding_length = learnable_length + pretrained_length
    assert embedding_length > 0, '0-size embeddings'

    return pretrained_embeddings, learnable_embeddings, embedding_length


def embed( model, input):
    input_list = []
    if model.pretrained_embeddings:
        input_list.append(model.pretrained_embeddings(input))
    if model.learnable_embeddings:
        input_list.append(model.learnable_embeddings(input))
    return torch.cat(tensors=input_list, dim=2)


def embedding_dropout( input, drop_range, p_drop=0.5, training=True):
    if p_drop > 0 and training and drop_range is not None:
        p = p_drop
        drop_from, drop_to = drop_range
        m = drop_to - drop_from     #length of the supervised embedding
        l = input.shape[2]          #total embedding length
        corr = (1 - p)
        input[:, :, drop_from:drop_to] = corr * F.dropout(input[:, :, drop_from:drop_to], p=p)
        input /= (1 - (p * m / l))

    return input
