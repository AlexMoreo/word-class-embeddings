import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
from tqdm import tqdm
import torch
from scipy.sparse import vstack, issparse


def index(data, vocab, known_words, analyzer, unk_index, out_of_vocabulary):
    """
    Index (i.e., replaces word strings with numerical indexes) a list of string documents
    :param data: list of string documents
    :param vocab: a fixed mapping [str]->[int] of words to indexes
    :param known_words: a set of known words (e.g., words that, despite not being included in the vocab, can be retained
    because they are anyway contained in a pre-trained embedding set that we know in advance)
    :param analyzer: the preprocessor in charge of transforming the document string into a chain of string words
    :param unk_index: the index of the 'unknown token', i.e., a symbol that characterizes all words that we cannot keep
    :param out_of_vocabulary: an incremental mapping [str]->[int] of words to indexes that will index all those words that
    are not in the original vocab but that are in the known_words
    :return:
    """
    indexes=[]
    vocabsize = len(vocab)
    unk_count = 0
    knw_count = 0
    out_count = 0
    pbar = tqdm(data, desc=f'indexing documents')
    for text in pbar:
        words = analyzer(text)
        index = []
        for word in words:
            if word in vocab:
                idx = vocab[word]
            else:
                if word in known_words:
                    if word not in out_of_vocabulary:
                        out_of_vocabulary[word] = vocabsize+len(out_of_vocabulary)
                    idx = out_of_vocabulary[word]
                    out_count += 1
                else:
                    idx = unk_index
                    unk_count += 1
            index.append(idx)
        indexes.append(index)
        knw_count += len(index)
        pbar.set_description(f'[unk = {unk_count}/{knw_count}={(100.*unk_count/knw_count):.2f}%]'
                             f'[out = {out_count}/{knw_count}={(100.*out_count/knw_count):.2f}%]')
    return indexes

# def stats(index_list):
#     lenghts = []
#     for indexes in tqdm(index_list):
#         lenghts.extend([len(indexes) for text,_ in data[domain][collection]])
#
#     mu,std = np.mean(lenghts),np.std(lenghts)
#     print(f'mean={mu:.3f}')
#     print(f'std={std:.3f}')
#     print(f'max={np.max(lenghts):.0f}')
#     print(f'min={np.min(lenghts):.0f}')
#     return mu,std

def define_pad_length(index_list):
    lengths = [len(index) for index in index_list]
    return int(np.mean(lengths)+np.std(lengths))

def pad(index_list, pad_index, max_pad_length=None):
    pad_length = np.max([len(index) for index in index_list])
    if max_pad_length is not None:
        pad_length = min(pad_length, max_pad_length)
    for i,indexes in enumerate(index_list):
        index_list[i] = [pad_index]*(pad_length-len(indexes)) + indexes[:pad_length]
    return index_list

def get_word_list(word2index1, word2index2=None): #TODO: redo
    def extract_word_list(word2index):
        return [w for w,i in sorted(word2index.items(), key=lambda x: x[1])]
    word_list = extract_word_list(word2index1)
    if word2index2 is not None:
        word_list += extract_word_list(word2index2)
    return word_list

def batchify(index_list, labels, batchsize, pad_index, target_long=False, max_pad_length=500):
    nsamples = len(index_list)
    nbatches = nsamples // batchsize + 1*(nsamples%batchsize>0)
    for b in range(nbatches):
        batch = index_list[b*batchsize:(b+1)*batchsize]
        batch_labels = labels[b*batchsize:(b+1)*batchsize]
        if issparse(batch_labels):
            batch_labels = batch_labels.toarray()
        batch = pad(batch, pad_index=pad_index, max_pad_length=max_pad_length)
        batch = torch.LongTensor(batch)
        totype = torch.LongTensor if target_long else torch.FloatTensor
        target = totype(batch_labels)
        yield batch.cuda(), target.cuda()

def batchify_unlabelled(index_list, batchsize, pad_index, max_pad_length=500):
    nsamples = len(index_list)
    nbatches = nsamples // batchsize + 1*(nsamples%batchsize>0)
    for b in range(nbatches):
        batch = index_list[b*batchsize:(b+1)*batchsize]
        batch = pad(batch, pad_index=pad_index, max_pad_length=max_pad_length)
        batch = torch.LongTensor(batch)
        yield batch.cuda()

def clip_gradient(model, clip_value=1e-1):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def predict(logits, classification_type='singlelabel'):
    if classification_type == 'multilabel':
        prediction = torch.sigmoid(logits) > 0.5
    elif classification_type == 'singlelabel':
        prediction = torch.argmax(logits, dim=1).view(-1, 1)
    else:
        print('unknown classification type')

    return prediction.detach().cpu().numpy()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)






