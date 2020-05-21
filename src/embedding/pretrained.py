from abc import ABC, abstractmethod
import torch, torchtext
import gensim
import os
import numpy as np

AVAILABLE_PRETRAINED = ['glove', 'word2vec', 'fasttext']

class KeyedVectors:

    def __init__(self, word2index, weights):
        assert len(word2index)==weights.shape[0], 'wrong number of dimensions'
        index2word = {i:w for w,i in word2index.items()}
        assert len([i for i in range(len(index2word)) if i not in index2word])==0, 'gaps in indexing not allowed'
        self.word2index = word2index
        self.index2word = index2word
        self.weights = weights

    def extract(self, words):
        dim = self.weights.shape[1]
        v_size = len(words)

        source_idx, target_idx = [], []
        for i,word in enumerate(words):
            if word not in self.word2index:
                continue
            j = self.word2index[word]
            source_idx.append(i)
            target_idx.append(j)

        extraction = np.zeros((v_size, dim))
        extraction[np.asarray(source_idx)] = self.weights[np.asarray(target_idx)]

        return extraction



class PretrainedEmbeddings(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def vocabulary(self): pass

    @abstractmethod
    def dim(self): pass

    @classmethod
    def reindex(cls, words, word2index):
        source_idx, target_idx = [], []
        for i, word in enumerate(words):
            if word not in word2index: continue
            j = word2index[word]
            source_idx.append(i)
            target_idx.append(j)
        source_idx = np.asarray(source_idx)
        target_idx = np.asarray(target_idx)
        return source_idx, target_idx


class GloVe(PretrainedEmbeddings):

    def __init__(self, setname='840B', path='./vectors_cache', max_vectors=None):
        super().__init__()
        print(f'Loading GloVe pretrained vectors from torchtext')
        self.embed = torchtext.vocab.GloVe(setname, cache=path, max_vectors=max_vectors)
        print('Done')

    def vocabulary(self):
        return set(self.embed.stoi.keys())

    def dim(self):
        return self.embed.dim

    def extract(self, words):
        source_idx, target_idx = PretrainedEmbeddings.reindex(words, self.embed.stoi)
        extraction = torch.zeros((len(words), self.dim()))
        extraction[source_idx] = self.embed.vectors[target_idx]
        return extraction


class Word2Vec(PretrainedEmbeddings):

    def __init__(self, path, limit=None, binary=True):
        super().__init__()
        print(f'Loading word2vec format pretrained vectors from {path}')
        assert os.path.exists(path), print(f'pre-trained keyed vectors not found in {path}')
        self.embed = gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary, limit=limit)
        self.word2index = {w: i for i,w in enumerate(self.embed.index2word)}
        print('Done')


    def vocabulary(self):
        return set(self.word2index.keys())

    def dim(self):
        return self.embed.vector_size

    def extract(self, words):
        source_idx, target_idx = PretrainedEmbeddings.reindex(words, self.word2index)
        extraction = np.zeros((len(words), self.dim()))
        extraction[source_idx] = self.embed.vectors[target_idx]
        extraction = torch.from_numpy(extraction).float()
        return extraction


class FastTextEmbeddings(Word2Vec):

    def __init__(self, path, limit=None):
        super().__init__(path, limit, binary=False)

