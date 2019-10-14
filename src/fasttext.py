import argparse
from sklearn.model_selection import train_test_split
from data.dataset import *
from embedding.pretrained import GloVe
from embedding.supervised import get_supervised_embeddings
from util.csv_log import CSVLog
from util.file import create_if_not_exist
from util.metrics import *
from time import time
from fastText import train_supervised
import io
import numpy as np
import torch
import torch.nn as nn


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def load_metadata_from_vectors_file(fname):
    with io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, d = map(int, fin.readline().split())
    return n,d

def save_vectors(fname, vocabulary, weights):
    n,d=weights.shape
    assert len(vocabulary) == n, 'vocabulary size and number of vectors do not match'

    with io.open(fname, 'w', encoding='utf-8', errors='ignore') as fin:
        fin.write(f'{n}\t{d}\n')
        for i,word in tqdm(enumerate(vocabulary), desc=f'[saving matrix of shape={weights.shape}] into {fname}'):
            vals_str = ' '.join(f'{x:.5f}' for x in weights[i].flat)
            # vals_str=np.array2string(weights[i], separator='\t', precision=5)[1:-1]
            fin.write(f'{word}\t{vals_str}\n')
    print('[done]')


def get_embedding_matrix_path(datasetname, dataset, pretrained, supervised, random, vec_matrix_path):
    matrix_name = f'{datasetname}-pretrained{pretrained}-supervised{supervised}-random{random}.vec'
    matrix_path = f'{vec_matrix_path}/{matrix_name}'
    if not os.path.exists(matrix_path):
        vocabulary, matrix = embedding_matrix(dataset, pretrained, supervised, random)
        create_if_not_exist(vec_matrix_path)
        save_vectors(matrix_path, vocabulary, matrix)
        dims = matrix.shape[1]
    else:
        _,dims = load_metadata_from_vectors_file(matrix_path)
    return matrix_path, dims


def embedding_matrix(dataset, pretrained=False, supervised=False, random=0):
    assert pretrained or supervised, 'useless call without requiring pretrained and/or supervised embeddings'
    vocabulary = dataset.vocabulary
    vocabulary = np.asarray(list(zip(*sorted(vocabulary.items(), key=lambda x:x[1])))[0])

    print('[embedding matrix]')
    pretrained_embeddings = []
    if pretrained:
        print('\t[pretrained-matrix: GloVe]')
        pretrained = GloVe()
        weights = pretrained.extract(vocabulary).numpy()
        pretrained_embeddings.append(weights)
        del pretrained

    if supervised:
        print('\t[supervised-matrix]')
        Xtr, _ = dataset.vectorize()
        Ytr = dataset.devel_labelmatrix
        F = get_supervised_embeddings(Xtr, Ytr)
        pretrained_embeddings.append(F)

    if random>0:
        print('\t[random-vectors]')
        R = torch.rand(len(vocabulary), random)
        R = nn.init.xavier_uniform_(R).numpy()
        pretrained_embeddings.append(R)

    pretrained_embeddings = np.hstack(pretrained_embeddings)
    print(f'[embedding matrix done] of shape={pretrained_embeddings.shape}')

    return vocabulary, pretrained_embeddings



def main():

    # init the log-file
    method_name = 'fasttext'
    method_name += '-bigrams' if args.bigrams else '-unigrams'
    method_name += '-glove' if args.pretrained else ''
    method_name += '-rand' if args.pretrained and args.learnable>0 else ''
    method_name += '-sup' if args.supervised else ''
    assert not(args.pretrained and args.supervised and args.learnable>0), 'this experiment was not supposed to be run!'
    logfile = CSVLog(args.log_file, ['dataset', 'method', 'lr', 'learnable', 'nepochs', 'seed', 'measure', 'value', 'timelapse'], autoflush=True)
    logfile.set_default('dataset', args.dataset)
    logfile.set_default('method', method_name)
    logfile.set_default('seed', args.seed)
    logfile.set_default('lr', args.lr)
    logfile.set_default('learnable', args.learnable)
    logfile.set_default('nepochs', args.nepochs)
    assert args.force or not logfile.already_calculated(), f'results for dataset {args.dataset} method {method_name} and run {args.seed} already calculated'

    # load dataset
    dataset = Dataset.load(dataset_name=args.dataset, pickle_path=args.pickle_path)

    matrix_path = None
    if args.pretrained or args.supervised:
        matrix_path, dims = get_embedding_matrix_path(args.dataset, dataset, args.pretrained, args.supervised, args.learnable, args.vec_matrix_path)



    analyzer = dataset.analyzer()
    devel = [' '.join(analyzer(t)) for t in tqdm(dataset.devel_raw, desc='indexing-devel')]
    test = [' '.join(analyzer(t)) for t in tqdm(dataset.test_raw, desc='indexing-test')]

    # dataset split tr/val/test
    val_size = min(int(len(devel) * .2), 20000)
    train, val, ytr, yva = train_test_split(devel, dataset.devel_target, test_size=val_size, random_state=args.seed, shuffle=True)
    yte = dataset.test_target
    print(f'tr={len(train)} va={len(val)} test={len(test)} docs')

    create_if_not_exist(args.dataset_dir)
    trainpath = get_input_file(train, ytr)

    loss = 'ova' if dataset.classification_type=='multilabel' else 'softmax'
    ngrams = 2 if args.bigrams else 1
    tinit = time()
    if matrix_path is None:
        model = train_supervised(
            input=trainpath, epoch=args.nepochs, lr=args.lr, wordNgrams=ngrams, verbose=2, minCount=1, loss=loss, dim=args.learnable
        )
    else:
        model = train_supervised(
            input=trainpath, epoch=args.nepochs, lr=args.lr, wordNgrams=ngrams, verbose=2, minCount=1, loss=loss, pretrainedVectors=matrix_path, dim=dims
        )
    tend = time()-tinit

    predic_and_eval(model, val, yva, 'va', dataset.classification_type, logfile, tend)
    predic_and_eval(model, test, yte, 'te', dataset.classification_type, logfile, tend)


def predic_and_eval(model, x, y, metric_prefix, classification_type, logfile, tend):
    if classification_type == 'multilabel':
        y_ = lil_matrix(y.shape)
        for i,t in tqdm(enumerate(x)):
            pred = model.f.predict(f'{t}\n', -1, 0.5, 'strict') #if any, returns a list of (probability, label)
            pred = np.array([int(l.replace('__label__','')) for prob,l in pred]) # take the index of the label
            y_[i,pred]=1
        y_ = y_.tocsr()
    else:
        y_=np.zeros(len(x), dtype=int)
        for i,t in tqdm(enumerate(x)):
            pred = model.f.predict(f'{t}\n', 1, 0, 'strict') #get the most probable label
            prob,label = pred[0]
            y_[i] = int(label.replace('__label__',''))

    Mf1, mf1, acc = evaluation(y, y_, classification_type)
    print(f'[{metric_prefix}] Macro-f1={Mf1:.3f} Micro-f1={mf1:.3f} Accuracy={acc:.3f} [took {tend}s]')
    logfile.add_row(measure=f'{metric_prefix}-macro-F1', value=Mf1, timelapse=tend)
    logfile.add_row(measure=f'{metric_prefix}-micro-F1', value=mf1, timelapse=tend)
    logfile.add_row(measure=f'{metric_prefix}-accuracy', value=acc, timelapse=tend)


def as_fasttext_labels(y):
    labels=[]
    if len(y.shape)==1:
        return [f'__label__{l}' for l in y]
    elif len(y.shape)==2:
        assert issparse(y), 'wrong format'
        for yi in y:
            labels.append(' '.join([f'__label__{l}' for l in yi.nonzero()[1]]))
    return labels


def get_input_file(texts, labels):
    file = f'{args.dataset_dir}/{args.dataset}-{len(texts)}-seed{args.seed}.train'
    if os.path.exists(file):
        print(f'file {file} already generated... skipping')
    else:
        print(f'generating input file {file}')
        labels = as_fasttext_labels(labels)
        with open(file, 'w') as fo:
            for i, doc in tqdm(enumerate(texts)):
                line = f'{labels[i]} {doc}\n'
                fo.write(line)
    return file


if __name__ == '__main__':
    available_datasets = Dataset.dataset_available

    # Training settings
    parser = argparse.ArgumentParser(description='Text Classification with Embeddings')
    parser.add_argument('--dataset', type=str, default='20newsgroups', metavar='N', help=f'dataset, one in {available_datasets}')
    parser.add_argument('--nepochs', type=int, default=200, metavar='N', help='number of epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=1, metavar='LR', help='learning rate (default: 1)')
    parser.add_argument('--learnable', type=int, default=100, metavar='N', help='dimension of the learnable embeddings (default 100)')
    parser.add_argument('--log-file', type=str, default='../log/log.csv', metavar='N', help='path to the log csv file')
    parser.add_argument('--dataset-dir', type=str, default='../fasttext/dataset', metavar='N', help='path to the directory where training files will be dumped')
    parser.add_argument('--vec-matrix-path', type=str, default='../fasttext/pretrained', metavar='N', help='path to the directory where training files will be dumped')
    parser.add_argument('--pickle-path', type=str, default=None, metavar='N', help=f'if set, specifies the path where to save/load the dataset pickled')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--bigrams', action='store_true', default=False, help='set word n-grams to 2, bigrams (default: 1, unigrams)')
    parser.add_argument('--force', action='store_true', default=False, help='do not check if this experiment has already been run')
    parser.add_argument('--pretrained', action='store_true', default=False, help='use GloVe pretrained embeddings')
    parser.add_argument('--supervised', action='store_true', default=False, help='use supervised embeddings')

    args = parser.parse_args()
    assert args.dataset in available_datasets, f'unknown dataset {args.dataset}'

    main()
