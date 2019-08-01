from data.dataset import *
from data.wipo_reader import WipoGammaDocument
from util.csv_log import CSVLog
import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from util.file import create_if_not_exist


outpath='../dataset_info'
create_if_not_exist(outpath)

stats = CSVLog(f'{outpath}/stats.csv',
               columns=['dataset', 'ndocs', 'nTr', 'nTe', 'nfeats', 'ncats', 'nwords', 'nunique',
                        'prev_mean', 'prev_std', 'prev_min', 'prev_max'],
               overwrite=False)

for dataset_name in ['reuters21578', 'rcv1', 'jrcall', 'ohsumed', '20newsgroups']:

    # load dataset
    dataset = Dataset.load(dataset_name=dataset_name, pickle_path=f'../pickles/{dataset_name}.pickle')

    print('vectorizing')
    Xtr, Xte = dataset.vectorize()
    nDtr, nF = Xtr.shape
    nDte = Xte.shape[0]
    nD = nDtr+nDte
    nC = dataset.nC

    print('counting words')
    analyzer = dataset.analyzer()
    alldocs = dataset.devel_raw + dataset.test_raw
    uniquewords = set()
    nwords = 0
    for doc in tqdm(alldocs, 'analyzing words'):
        doc_words = analyzer(doc)
        uniquewords.update(doc_words)
        nwords += len(doc_words)
    nunique = len(uniquewords)

    prevalences = dataset.devel_labelmatrix.sum(axis=0)

    stats.add_row(dataset=dataset_name, ndocs=nD, nTr=nDtr, nTe=nDte, nfeats=nF, ncats=nC,
                  nwords=nwords, nunique=nunique,
                  prev_mean=prevalences.mean(), prev_std=prevalences.std(),
                  prev_min=prevalences.min(), prev_max=prevalences.max())

    print(stats.df)

    # n, bins, patches = plt.hist(prevalences, density=False, facecolor='g', alpha=0.75)
    # plt.xlabel('Prevalences')
    # plt.ylabel('count')
    # plt.title(f'Distribution of class prevalence in {dataset_name}')
    # # plt.axis([40, 160, 0, 0.03])
    # plt.grid(True)
    # plt.savefig(f'{outpath}/{dataset_name}.png')
    # plt.close('all')



