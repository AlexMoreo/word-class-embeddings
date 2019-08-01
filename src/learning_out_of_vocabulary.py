import argparse
import torch
import torchtext
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import scipy
from baselines import svm_performance
from data.tsr_function__ import round_robin_selection
from embedding.supervised import get_supervised_embeddings, fit_predict, multi_domain_sentiment_embeddings
from main import index_dataset
from model.cnn_class import CNN
from model.embedding_predictor import EmbeddingPredictor
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
    fullpickle = f'../pickles/{opt.dataset}-index.pickle'
    pretrained = GloVe()
    dataset = Dataset.load(dataset_name=opt.dataset, pickle_path=f'../pickles/{opt.dataset}.pickle').show()
    if os.path.exists(fullpickle):
        print(f'loaded pre-computed index from {fullpickle}')
        word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = pickle.load(open(fullpickle, 'rb'))
    else:
        word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(dataset, pretrained)

    print(f'vocabulary-size={len(word2index)}')
    print(f'out-of-voc-size={len(out_of_vocabulary)}')

    print('[embedding matrix]')

    in_vocabulary_words = get_word_list(word2index)
    out_vocabulary_words = get_word_list(out_of_vocabulary)
    word_list = in_vocabulary_words + out_vocabulary_words
    in_vocabulary_words = np.asarray(in_vocabulary_words)
    out_vocabulary_words = np.asarray(out_vocabulary_words)
    weights = pretrained.extract(word_list)
    del pretrained

    print('\t[supervised-matrix]')
    # v = CountVectorizer()
    # Xte = v.fit_transform(dataset.test_raw, mindf=1)
    Xtr, Xte = dataset.vectorize()
    Ytr = dataset.devel_labelmatrix
    Yte = dataset.test_labelmatrix
    nC=Ytr.shape[1]
    F_in_vocab = get_supervised_embeddings(Xtr, Ytr)

    V1,E1=weights.shape
    V2,E2=F_in_vocab.shape
    e = EmbeddingPredictor(input_size=E1, output_size=E2).cuda()
    e.fit(weights[:V2], F_in_vocab)
    F_out_vocab = e.predict(weights[V2:])

    F_in_vocab = torch.from_numpy(F_in_vocab).float()
    F_out_vocab = torch.from_numpy(F_out_vocab).float()

    print(f'pretrained-shape={weights.shape}')
    print(f'supervised-shape={F_in_vocab.shape}')
    print(f'learnt-shape={F_out_vocab.shape}')

    idx, vals, cats = round_robin_selection(Xtr, Ytr, k=nC)
    F_in_vocab = F_in_vocab[idx]
    in_vocabulary_words = in_vocabulary_words[idx]
    F_in_vocab = torch.nn.functional.normalize(F_in_vocab, p=2, dim=1)
    F_out_vocab = torch.nn.functional.normalize(F_out_vocab, p=2, dim=1)
    new_correlations = torch.mm(F_out_vocab,torch.t(F_in_vocab))
    print(f'correlations-shape={new_correlations.shape}')
    # cols = new_correlations.shape[1]

    argsorted = torch.argsort(new_correlations, dim=0, descending=True)[:10,:]
    # top_rows = (argsorted // cols).numpy().tolist()
    # top_cols = (argsorted % cols).numpy().tolist()
    argsorted=torch.t(argsorted)
    for i,idx_i in enumerate(argsorted):
        similar_str = ""
        for idx_ij in idx_i:
            word = out_vocabulary_words[idx_ij]
            corr = new_correlations[idx_ij, i]
            similar_str += f'{word} ({corr:.3f}), '
        print(f'{in_vocabulary_words[i]} (cat={cats[i]}): {similar_str}')

    # argsorted = torch.argsort(new_correlations.view(-1), descending=True)[:100]
    # top_rows = (argsorted // cols).numpy().tolist()
    # top_cols = (argsorted % cols).numpy().tolist()
    # for i,(r,c) in enumerate(zip(top_rows,top_cols)):
    #     print(f'{out_vocabulary_words[r]} --> {in_vocabulary_words[c]} ({new_correlations[r,c]:.3f})')



    print('[done]')





if __name__ == '__main__':
    available_datasets = Dataset.dataset_available

    # Training settings
    parser = argparse.ArgumentParser(description='Text Classification with Embeddings')
    parser.add_argument('--dataset', type=str, default='rcv1', metavar='N', help=f'dataset, one in {available_datasets}')

    opt = parser.parse_args()

    assert torch.cuda.is_available(), 'CUDA not available'
    device = torch.device('cuda')
    torch.manual_seed(43)

    assert opt.dataset in available_datasets, f'unknown dataset {opt.dataset}'

    main()
