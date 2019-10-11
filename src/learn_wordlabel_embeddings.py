import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from embedding.supervised import get_supervised_embeddings
from time import time
import numpy as np

def read_dataset():
    all_sentences, all_labels = [], []
    for line in tqdm(open(opt.input, 'rt').readlines(), f'loading {opt.input}'):
        sentence, labels = extract_labels(line)
        if sentence:
            all_sentences.append(sentence)
            all_labels.append(labels)
    return all_sentences, all_labels


def extract_labels(x):
    sentence=[]
    labels=[]
    for term in x.strip().split():
        addto = labels if term.startswith(opt.label) else sentence
        addto.append(term)
    sentence = ' '.join(sentence)
    return sentence, labels


def vectorize_text(sentences):
    vectorizer = TfidfVectorizer(min_df=opt.minfreq, sublinear_tf=True)
    X = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names()
    return X, features


def vectorize_labels(labels):
    mlb = MultiLabelBinarizer(sparse_output=True)
    Y = mlb.fit_transform(labels)
    return Y


def compute_embeddings(sentences, labels):
    tinit = time()
    X, features = vectorize_text(sentences)
    Y = vectorize_labels(labels)
    bow_time = time() - tinit
    S = get_supervised_embeddings(X, Y, max_label_space=opt.maxdim,  method=opt.method)
    sup_time = time() - bow_time - tinit
    total_time = bow_time + sup_time

    (nD, nF), nC = X.shape, Y.shape[1]
    print(f'documents={nD} features={nF} categories={nC}, word-class embedding matrix shape={S.shape}')
    print(f'timming:\n'
          f'\ttotal={total_time:.2f}s\n'
          f'\tcompute bow={bow_time:.2f}s ({100*bow_time/total_time:.2f}%)\n'
          f'\tcompute word-class embeddings={sup_time:.2f}s ({100*sup_time/total_time:.2f}%)')

    return S, features


def write_embeddings(S, features, format='text'):
    if format=='text':
        with open(opt.output, 'wt') as foo:
            for i,term in tqdm(enumerate(features), f'saving matrix in {opt.output}'):
                str_vec = np.array2string(S[i],precision=5,max_line_width=np.nan)[2:-2]
                foo.write(f'{term} {str_vec}\n')


def main():
    sentences, labels = read_dataset()
    S, features = compute_embeddings(sentences, labels)
    write_embeddings(S, features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn Word-Class Embeddings from dataset in fastText format')
    parser.add_argument('-o', '--output', help='Output file name', default='word-class-embeddings.txt')
    parser.add_argument('-m', '--method', help='Correlation method', default='dotn')
    parser.add_argument('-f', '--minfreq', help='Minimum number of occurrences of terms', default=5)
    parser.add_argument('-d', '--maxdim', help='Maximum number of dimensions (if there are more categories, then PCA is applied)', default=300)
    parser.add_argument('-l', '--label', help='Label prefix (default __label__)', default='__label__')
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-i', '--input', help='Input file path', required=True, type=str)
    opt = parser.parse_args()

    main()
