import os
import pickle
import tarfile
from os.path import join
# from sklearn.externals.six.moves import urllib
import urllib.request

from data.labeled import LabelledDocuments
from util.file import create_if_not_exist, download_file_if_not_exists

import math

def fetch_ohsumed50k(data_path=None, subset='train', train_test_split=0.7):
    _dataname = 'ohsumed50k'
    if data_path is None:
        data_path = join(os.path.expanduser('~'), _dataname)
    create_if_not_exist(data_path)

    pickle_file = join(data_path, _dataname + '.' + subset + str(train_test_split) + '.pickle')
    if not os.path.exists(pickle_file):
        DOWNLOAD_URL = ('http://disi.unitn.it/moschitti/corpora/ohsumed-all-docs.tar.gz')
        archive_path = os.path.join(data_path, 'ohsumed-all-docs.tar.gz')
        download_file_if_not_exists(DOWNLOAD_URL, archive_path)
        untardir = 'ohsumed-all'
        if not os.path.exists(os.path.join(data_path, untardir)):
            print("untarring ohsumed...")
            tarfile.open(archive_path, 'r:gz').extractall(data_path)

        target_names = []
        doc_classes = dict()
        class_docs = dict()
        content = dict()
        doc_ids = set()
        for cat_id in os.listdir(join(data_path, untardir)):
            target_names.append(cat_id)
            class_docs[cat_id] = []
            for doc_id in os.listdir(join(data_path, untardir, cat_id)):
                doc_ids.add(doc_id)
                text_content = open(join(data_path, untardir, cat_id, doc_id), 'r').read()
                if doc_id not in doc_classes: doc_classes[doc_id] = []
                doc_classes[doc_id].append(cat_id)
                if doc_id not in content: content[doc_id] = text_content
                class_docs[cat_id].append(doc_id)
        target_names.sort()
        print('Read %d different documents' % len(doc_ids))

        splitdata = dict({'train': [], 'test': []})
        for cat_id in target_names:
            free_docs = [d for d in class_docs[cat_id] if (d not in splitdata['train'] and d not in splitdata['test'])]
            if len(free_docs) > 0:
                split_point = int(math.floor(len(free_docs) * train_test_split))
                splitdata['train'].extend(free_docs[:split_point])
                splitdata['test'].extend(free_docs[split_point:])
        for split in ['train', 'test']:
            dataset = LabelledDocuments([], [], target_names)
            for doc_id in splitdata[split]:
                dataset.data.append(content[doc_id])
                dataset.target.append([target_names.index(cat_id) for cat_id in doc_classes[doc_id]])
            pickle.dump(dataset,
                        open(join(data_path, _dataname + '.' + split + str(train_test_split) + '.pickle'), 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)

    print(pickle_file)
    return pickle.load(open(pickle_file, 'rb'))
