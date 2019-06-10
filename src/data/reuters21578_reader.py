# Modified version of the code originally implemented by Eustache Diemert <eustache@diemert.fr>
#          @FedericoV <https://github.com/FedericoV/>
# with License: BSD 3 clause

import os.path
import re
import tarfile
from sklearn.datasets import get_data_home
from sklearn.externals.six.moves import html_parser
from sklearn.externals.six.moves import urllib
import pickle
from glob import glob
import numpy as np
from data.labeled import LabelledDocuments


def _not_in_sphinx():
    # Hack to detect whether we are running by the sphinx builder
    return '__file__' in globals()

class ReutersParser(html_parser.HTMLParser):
    """Utility class to parse a SGML file and yield documents one at a time."""

    def __init__(self, encoding='latin-1', data_path=None):
        self.data_path = data_path
        self.download_if_not_exist()
        self.tr_docs = []
        self.te_docs = []
        html_parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding
        self.empty_docs = 0

    def handle_starttag(self, tag, attrs):
        method = 'start_' + tag
        getattr(self, method, lambda x: None)(attrs)

    def handle_endtag(self, tag):
        method = 'end_' + tag
        getattr(self, method, lambda: None)()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.in_unproc_text = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""
        self.text = ""

    def parse(self, fd):
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
        self.close()

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data
        elif self.in_unproc_text:
            self.text += data

    def start_reuters(self, attributes):
        topic_attr = attributes[0][1]
        lewissplit_attr = attributes[1][1]
        self.lewissplit = u'unused'
        if topic_attr==u'YES':
            if lewissplit_attr == u'TRAIN':
                self.lewissplit = 'train'
            elif lewissplit_attr == u'TEST':
                self.lewissplit = 'test'
        pass

    def end_reuters(self):
        self.body = re.sub(r'\s+', r' ', self.body)
        if self.lewissplit != u'unused':
            parsed_doc = {'title': self.title, 'body': self.body, 'unproc':self.text, 'topics': self.topics}
            if (self.title+self.body+self.text).strip() == '':
                self.empty_docs += 1
            if self.lewissplit == u'train':
                self.tr_docs.append(parsed_doc)
            elif self.lewissplit == u'test':
                self.te_docs.append(parsed_doc)
        self._reset()

    def start_title(self, attributes):
        self.in_title = 1

    def end_title(self):
        self.in_title = 0

    def start_body(self, attributes):
        self.in_body = 1

    def end_body(self):
        self.in_body = 0

    def start_topics(self, attributes):
        self.in_topics = 1

    def end_topics(self):
        self.in_topics = 0

    def start_text(self, attributes):
        if len(attributes)>0 and attributes[0][1] == u'UNPROC':
            self.in_unproc_text = 1

    def end_text(self):
        self.in_unproc_text = 0

    def start_d(self, attributes):
        self.in_topic_d = 1

    def end_d(self):
        if self.in_topics:
            self.topics.append(self.topic_d)
        self.in_topic_d = 0
        self.topic_d = ""

    def download_if_not_exist(self):
        DOWNLOAD_URL = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
                        'reuters21578-mld/reuters21578.tar.gz')
        ARCHIVE_FILENAME = 'reuters21578.tar.gz'

        if self.data_path is None:
            self.data_path = os.path.join(get_data_home(), "reuters")
        if not os.path.exists(self.data_path):
            """Download the dataset."""
            print("downloading dataset (once and for all) into %s" % self.data_path)
            os.mkdir(self.data_path)

            def progress(blocknum, bs, size):
                total_sz_mb = '%.2f MB' % (size / 1e6)
                current_sz_mb = '%.2f MB' % ((blocknum * bs) / 1e6)
                if _not_in_sphinx():
                    print('\rdownloaded %s / %s' % (current_sz_mb, total_sz_mb), end='')

            archive_path = os.path.join(self.data_path, ARCHIVE_FILENAME)
            urllib.request.urlretrieve(DOWNLOAD_URL, filename=archive_path,
                                       reporthook=progress)
            if _not_in_sphinx():
                print('\r', end='')
            print("untarring Reuters dataset...")
            tarfile.open(archive_path, 'r:gz').extractall(self.data_path)
            print("done.")


def fetch_reuters21578(data_path=None, subset='train'):
    if data_path is None:
        data_path = os.path.join(get_data_home(), 'reuters21578')
    reuters_pickle_path = os.path.join(data_path, "reuters." + subset + ".pickle")
    if not os.path.exists(reuters_pickle_path):
        parser = ReutersParser(data_path=data_path)
        for filename in glob(os.path.join(data_path, "*.sgm")):
            parser.parse(open(filename, 'rb'))
        # index category names with a unique numerical code (only considering categories with training examples)
        tr_categories = np.unique(np.concatenate([doc['topics'] for doc in parser.tr_docs])).tolist()

        def pickle_documents(docs, subset):
            for doc in docs:
                doc['topics'] = [tr_categories.index(t) for t in doc['topics'] if t in tr_categories]
            pickle_docs = {'categories': tr_categories, 'documents': docs}
            pickle.dump(pickle_docs, open(os.path.join(data_path, "reuters." + subset + ".pickle"), 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
            return pickle_docs

        pickle_tr = pickle_documents(parser.tr_docs, "train")
        pickle_te = pickle_documents(parser.te_docs, "test")
        # self.sout('Empty docs %d' % parser.empty_docs)
        requested_subset = pickle_tr if subset == 'train' else pickle_te
    else:
        requested_subset = pickle.load(open(reuters_pickle_path, 'rb'))

    data = [(u'{title}\n{body}\n{unproc}'.format(**doc), doc['topics']) for doc in requested_subset['documents']]
    text_data, topics = zip(*data)
    return LabelledDocuments(data=text_data, target=topics, target_names=requested_subset['categories'])




if __name__=='__main__':
    reuters_train = fetch_reuters21578(subset='train')
    print(reuters_train.data)