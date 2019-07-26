import argparse
import torch
import torchtext
import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import scipy
from baselines import svm_performance
from data.tsr_function__ import get_tsr_matrix, get_supervised_matrix, positive_information_gain
from embedding.supervised import get_supervised_embeddings, fit_predict, multi_domain_sentiment_embeddings
from main import embedding_matrix, index_dataset
from model.cnn_class import CNN
from model.lstm_attn_class import AttentionModel
from model.lstm_class import LSTMClassifier
from util.early_stop import EarlyStopping
from common import *
from data.dataset import *
from util.csv_log import CSVLog
from util.file import create_if_not_exist
from util.metrics import *
from time import time
from embedding.pretrained import *

def main():

    # load dataset
    pretrained=None
    if opt.pretrained== 'glove':
        pretrained = GloVe()
    elif opt.pretrained== 'word2vec':
        pretrained = Word2Vec(path=opt.word2vec_path, limit=1000000)

    print('[loading dataset]')
    dataset = Dataset.load(dataset_name=opt.dataset, pickle_path=opt.pickle_path).show()
    word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(dataset, pretrained)

    print('[obtaining embeddings]')
    vocabsize = len(word2index) + len(out_of_vocabulary)
    pretrained_embeddings, sup_range = embedding_matrix(dataset, pretrained, vocabsize, word2index, out_of_vocabulary, opt)
    pret_range = [0,sup_range[0] if sup_range is not None else pretrained_embeddings.shape[1]] # range along dim=1 of unsupervised pretrained vectors
    index2word={i:w for w,i in word2index.items()}

    print('[vectorizing]')
    X,_ = dataset.vectorize()
    Y = dataset.devel_labelmatrix
    nC = Y.shape[1]

    print(f'[selectiong {opt.terms} terms]')
    FC = get_tsr_matrix(get_supervised_matrix(X, Y), positive_information_gain).T
    best_features_idx = np.argsort(-FC, axis=0).flatten()
    selected_indexes_set = set()
    selected_indexes = list()
    selected_class = list()
    round_robin = iter(best_features_idx)
    round=0
    skip=0
    while len(selected_indexes) < opt.terms:
        term_idx = next(round_robin)
        round=(round+1)%nC
        if term_idx not in selected_indexes_set:
            if opt.pretrained is not None:
                if pretrained_embeddings[term_idx,:pret_range[1]].sum()==0:
                    skip+=1
                    continue

            selected_indexes_set.add(term_idx)
            selected_indexes.append(term_idx)
            selected_class.append(round)

    print(f'skipped = {skip}')
    selected_indexes = np.asarray(selected_indexes)
    selected_embeddings = pretrained_embeddings[selected_indexes]
    selected_terms = [index2word[idx] for idx in selected_indexes]

    name = ('-'+opt.pretrained if opt.pretrained is not None else '')
    name += ('-supervised' if opt.supervised else '')
    data = f'{opt.embedding_dir}/{opt.dataset}{name}-{opt.terms}.vec'
    meta = f'{opt.embedding_dir}/{opt.dataset}{name}-{opt.terms}.meta'
    with open(data,'wt') as data, open(meta,'wt') as meta:
        meta.write('Term\tClass\n')
        for emb, cat, term in zip(selected_embeddings, selected_class,selected_terms):
            emb_text = '\t'.join([f'{emb_i:.5f}' for emb_i in emb])
            data.write(f'{emb_text}\n')
            meta.write(f'{term}\t{cat}\n')
            assert len(term.strip())>0, 'empty word?'


if __name__ == '__main__':
    available_datasets = Dataset.dataset_available

    # Training settings
    parser = argparse.ArgumentParser(description='Text Classification with Embeddings')
    parser.add_argument('--dataset', type=str, default='20newsgroups', metavar='N', help=f'dataset, one in {available_datasets}')
    parser.add_argument('--embedding-dir', type=str, default='../embeddings', metavar='N', help='path to the embedding-dir where data for visualization will be dumped')
    parser.add_argument('--pretrained', type=str, default=None, metavar='N', help='pretrained embeddings, use "glove" or "word2vec" (default None)')
    parser.add_argument('--supervised', action='store_true', default=False, help='use supervised embeddings')
    parser.add_argument('--word2vec-path', type=str, default='../datasets/Word2Vec/GoogleNews-vectors-negative300.bin',
                        metavar='N', help=f'path to GoogleNews-vectors-negative300.bin pretrained vectors')
    parser.add_argument('--terms', type=int, default=1000, metavar='N', help=f'number of terms to extract')
    parser.add_argument('--pickle-dir', type=str, default='../pickles', metavar='N',
                        help=f'if set, specifies the path where to'
                        f'save/load the dataset pickled (set to None if you prefer not to retain the pickle file)')
    opt = parser.parse_args()

    assert torch.cuda.is_available(), 'CUDA not available'
    device = torch.device('cuda')

    assert opt.dataset in available_datasets, f'unknown dataset {opt.dataset}'
    assert opt.pretrained in {None, 'glove', 'word2vec'}, f'unknown pretrained set {opt.pretrained}'
    if opt.pickle_dir: opt.pickle_path = join(opt.pickle_dir, f'{opt.dataset}.pickle')
    create_if_not_exist(opt.embedding_dir)
    opt.sentiment=False
    assert opt.pretrained or opt.supervised, 'no embedding to visualize...'
    opt.supervised_method='dotn'
    opt.supervised_nozscore=False
    opt.predict_missing=opt.predict_all=False

    main()
