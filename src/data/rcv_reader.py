from zipfile import ZipFile
import xml.etree.ElementTree as ET
from data.labeled import LabelledDocuments
from util.file import list_files
from os.path import join, exists
from util.file import download_file_if_not_exists
import re
from collections import Counter

RCV1_TOPICHIER_URL = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a02-orig-topics-hierarchy/rcv1.topics.hier.orig"
RCV1_BASE_URL = "http://www.daviddlewis.com/resources/testcollections/rcv1/"

rcv1_test_data_gz = ['lyrl2004_tokens_test_pt0.dat.gz',
             'lyrl2004_tokens_test_pt1.dat.gz',
             'lyrl2004_tokens_test_pt2.dat.gz',
             'lyrl2004_tokens_test_pt3.dat.gz']

rcv1_train_data_gz = ['lyrl2004_tokens_train.dat.gz']

rcv1_doc_cats_data_gz = 'rcv1-v2.topics.qrels.gz'

class RCV_Document:
    def __init__(self, id, text, categories, date=''):
        self.id = id
        self.date = date
        self.text = text
        self.categories = categories

class IDRangeException(Exception): pass

nwords = []

def parse_document(xml_content, valid_id_range=None):
    root = ET.fromstring(xml_content)

    doc_id = root.attrib['itemid']
    if valid_id_range is not None:
        if not valid_id_range[0] <= int(doc_id) <= valid_id_range[1]:
            raise IDRangeException

    doc_categories = [cat.attrib['code'] for cat in
                      root.findall('.//metadata/codes[@class="bip:topics:1.0"]/code')]

    doc_date = root.attrib['date']
    doc_title = root.find('.//title').text
    doc_headline = root.find('.//headline').text
    doc_body = '\n'.join([p.text for p in root.findall('.//text/p')])

    if not doc_body:
        raise ValueError('Empty document')

    if doc_title is None: doc_title = ''
    if doc_headline is None or doc_headline in doc_title: doc_headline = ''
    text = '\n'.join([doc_title, doc_headline, doc_body]).strip()

    # text_length = len(text.split())
    # global nwords
    # nwords.append(text_length)
    # if text_length < min_words:
    #     raise IDRangeException

    return RCV_Document(id=doc_id, text=text, categories=doc_categories, date=doc_date)


def fetch_RCV1(data_path, subset='all'):

    assert subset in ['train', 'test', 'all'], 'split should either be "train", "test", or "all"'

    request = []
    labels = set()
    read_documents = 0

    training_documents = 23149
    test_documents = 781265

    if subset == 'all':
        split_range = (2286, 810596)
        expected = training_documents+test_documents
    elif subset == 'train':
        split_range = (2286, 26150)
        expected = training_documents
    else:
        split_range = (26151, 810596)
        expected = test_documents

    # global nwords
    # nwords=[]
    for part in list_files(data_path):
        if not re.match('\d+\.zip', part): continue
        target_file = join(data_path, part)
        assert exists(target_file), \
            "You don't seem to have the file "+part+" in " + data_path + ", and the RCV1 corpus can not be downloaded"+\
            " w/o a formal permission. Please, refer to " + RCV1_BASE_URL + " for more information."
        zipfile = ZipFile(target_file)
        for xmlfile in zipfile.namelist():
            xmlcontent = zipfile.open(xmlfile).read()
            try:
                doc = parse_document(xmlcontent, valid_id_range=split_range)
                labels.update(doc.categories)
                request.append(doc)
                read_documents += 1
            except (IDRangeException,ValueError) as e:
                pass
            print('\r[{}] read {} documents'.format(part, len(request)), end='')
            if read_documents == expected: break
        if read_documents == expected: break

    print()
    # print('ave:{} std {} min {} max {}'.format(np.mean(nwords), np.std(nwords), np.min(nwords), np.max(nwords)))

    return LabelledDocuments(data=[d.text for d in request], target=[d.categories for d in request], target_names=list(labels))



def fetch_topic_hierarchy(path, topics='all'):
    assert topics in ['all', 'leaves']

    download_file_if_not_exists(RCV1_TOPICHIER_URL, path)
    hierarchy = {}
    for line in open(path, 'rt'):
        parts = line.strip().split()
        parent,child = parts[1],parts[3]
        if parent not in hierarchy:
            hierarchy[parent]=[]
        hierarchy[parent].append(child)

    del hierarchy['None']
    del hierarchy['Root']
    print(hierarchy)

    if topics=='all':
        topics = set(hierarchy.keys())
        for parent in hierarchy.keys():
            topics.update(hierarchy[parent])
        return list(topics)
    elif topics=='leaves':
        parents = set(hierarchy.keys())
        childs = set()
        for parent in hierarchy.keys():
            childs.update(hierarchy[parent])
        return list(childs.difference(parents))


if __name__=='__main__':

    RCV1_PATH = '../../datasets/RCV1-v2/unprocessed_corpus'

    rcv1_train = fetch_RCV1(RCV1_PATH, subset='train')
    rcv1_test = fetch_RCV1(RCV1_PATH, subset='test')

    print('read {} documents in rcv1-train, and {} labels'.format(len(rcv1_train.data), len(rcv1_train.target_names)))
    print('read {} documents in rcv1-test, and {} labels'.format(len(rcv1_test.data), len(rcv1_test.target_names)))

    cats = Counter()
    for cats in rcv1_train.target: cats.update(cats)
    print('RCV1', cats)
