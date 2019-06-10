import os
import numpy as np
import tarfile
from util.file import *
from data.labeled import LabelledDocuments


"""
http://ai.stanford.edu/~amaas/data/sentiment/index.html

@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
"""

IMDB_URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

def fetch_IMDB(subset, data_home='../datasets/IMDB'):
    assert subset in ['train', 'test'], 'subset should either be "train" or "test"'
    data_path = os.path.join(data_home, 'aclImdb_v1')
    data_tar =  f'{data_path}.tar.gz'

    if not os.path.exists(data_path):
        download_file_if_not_exists(IMDB_URL, data_tar)
        tarfile.open(data_tar, 'r:gz').extractall(data_path)

    dataset = LabelledDocuments(data=[], target=[], target_names=['pos', 'neg'])
    for label in ['pos', 'neg']:
        path = f'{data_path}/aclImdb/{subset}/{label}'
        docs = [open(os.path.join(path, file)).read() for file in list_files(path)]
        dataset.data.extend(docs)
        dataset.target.extend([1 if label == 'pos' else 0]*len(docs))

    dataset.target = np.asarray(dataset.target)

    return dataset


def fetch_MDSunprocessed(DOWNLOAD_URL = 'https://www.cs.jhu.edu/~mdredze/datasets/sentiment/unprocessed.tar.gz',
                         dataset_home = '../datasets/MDSunprocessed',
                         min_pos=0, max_pos=-1,
                         min_neg=0, max_neg=-1,
                         min_unlabeled=0, max_unlabeled=50000,
                         domains='all'):
    """
    Fetchs the unprocessed version of the  Multi-Domain Sentiment Dataset (version 2.0) for cross-domain adaptation defined in:
    John Blitzer, Mark Dredze, Fernando Pereira.
    Biographies, Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification.
    Association of Computational Linguistics (ACL), 2007.
    """
    dataset_path = join(dataset_home, 'unprocessed.tar.gz')
    create_if_not_exist(dataset_home)

    if not exists(dataset_path):
        print("downloading multidomain dataset (once and for all) into %s" % dataset_path)
        download_file(DOWNLOAD_URL, dataset_path)
        print("untarring dataset...")
        tarfile.open(dataset_path, 'r:gz').extractall(dataset_home)

    dataset_path = join(dataset_home, 'sorted_data')
    check_counts = {dom.replace(' ','_'):int(count) for dom,count in [line.strip().split('\t') for line in open(join(dataset_path, 'summary.txt')).readlines()]}

    documents = {}
    for domain in list_dirs(dataset_path):
        if domains!='all' and domain not in domains: continue
        if check_counts.get(f'{domain}/unlabeled.review', 0) < min_unlabeled: continue
        if check_counts.get(f'{domain}/positive.review', 0) < min_pos: continue
        if check_counts.get(f'{domain}/negative.review', 0) < min_neg: continue

        documents[domain]={'labeled':[], 'unlabeled':[]}

        for type in ['positive.review','negative.review','unlabeled.review']:
            if type=='unlabeled.review' and max_unlabeled==0: continue

            with open(join(dataset_path,domain,type), errors='ignore') as fo:
                def nextline(fo): return next(fo).strip()
                docs = []
                try:
                    while True:
                        line = nextline(fo)
                        if line == '<review>':
                            label=text=None

                        if line == '<rating>':
                            rating = int(float(nextline(fo)))
                            if rating > 3: label=1
                            elif rating < 3: label=0
                            assert nextline(fo)=='</rating>', 'wrong format'

                        if line in ['<review_text>', '<title>']:
                            if text is None: text=[]
                            while True:
                                line = nextline(fo)
                                if line not in ['</review_text>','</title>']:
                                    if line: text.append(line)
                                else: break

                        if line == '</review>':
                            text = ' '.join(text)
                            if label is not None and text:
                                docs.append((text,label))
                                if type=='unlabeled.review' and max_unlabeled!=-1 and len(docs)>=max_unlabeled:
                                    raise StopIteration
                                if type=='positive.review' and max_pos!=-1 and len(docs)>=max_pos:
                                    raise StopIteration
                                if type=='negative.review' and max_neg!=-1 and len(docs)>=max_neg:
                                    raise StopIteration
                            else:
                                print('lost {} {} (in {})'.format(text, label, len(docs)))

                except StopIteration:
                    print("domain '{}' file {}: {} documents".format(domain, type, len(docs)))
                    expected = check_counts.get('/'.join((domain, type)), 0)
                    if type == 'unlabeled.review' and expected > max_unlabeled: expected = max_unlabeled
                    found = len(docs)
                    assert found == expected, 'count check error for {}/{} (found {}, expected {})'.format(
                        domain, type, found, expected)

            documents[domain]['unlabeled' if type == 'unlabeled.review' else 'labeled'].extend(docs)

    return documents
