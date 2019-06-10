import os, sys
from os.path import join
import tarfile
import xml.etree.ElementTree as ET
from sklearn.datasets import get_data_home
import pickle
import rdflib
from rdflib.namespace import RDF, SKOS
from rdflib import URIRef
import zipfile
from collections import Counter
from tqdm import tqdm
from random import shuffle
from util.file import *


class JRCAcquis_Document:
    def __init__(self, id, name, lang, year, head, body, categories):
        self.id = id
        self.parallel_id = name
        self.lang = lang
        self.year = year
        self.text = body if not head else head + "\n" + body
        self.categories = categories

    @classmethod
    def get_text(cls, jrc_documents):
        return [d.text for d in jrc_documents]

    @classmethod
    def get_target(cls, jrc_documents):
        return [d.categories for d in jrc_documents]


# this is a workaround... for some reason, acutes are codified in a non-standard manner in titles
# however, it seems that the title is often appearing as the first paragraph in the text/body (with
# standard codification), so it might be preferable not to read the header after all (as here by default)
def _proc_acute(text):
    for ch in ['a','e','i','o','u']:
        text = text.replace('%'+ch+'acute%',ch)
    return text

def parse_document(file, year, head=False):
    root = ET.parse(file).getroot()

    doc_name = root.attrib['n'] # e.g., '22006A0211(01)'
    doc_lang = root.attrib['lang'] # e.g., 'es'
    doc_id   = root.attrib['id'] # e.g., 'jrc22006A0211_01-es'
    doc_categories = [cat.text for cat in root.findall('.//teiHeader/profileDesc/textClass/classCode[@scheme="eurovoc"]')]
    doc_head = _proc_acute(root.find('.//text/body/head').text) if head else ''
    doc_body = '\n'.join([p.text for p in root.findall('.//text/body/div[@type="body"]/p')])

    def raise_if_empty(field, from_file):
        if isinstance(field, str):
            if not field.strip():
                raise ValueError("Empty field in file %s" % from_file)

    raise_if_empty(doc_name, file)
    raise_if_empty(doc_lang, file)
    raise_if_empty(doc_id, file)
    if head: raise_if_empty(doc_head, file)
    raise_if_empty(doc_body, file)

    return JRCAcquis_Document(id=doc_id, name=doc_name, lang=doc_lang, year=year, head=doc_head, body=doc_body, categories=doc_categories)

#filters out documents which do not contain any category in the cat_filter list, and filter all labels not in cat_filter
def _filter_by_category(doclist, cat_filter):
    if not isinstance(cat_filter, frozenset):
        cat_filter = frozenset(cat_filter)
    filtered = []
    for doc in doclist:
        doc.categories = list(cat_filter & set(doc.categories))
        if doc.categories:
            doc.categories.sort()
            filtered.append(doc)
    print("filtered %d documents out without categories in the filter list" % (len(doclist) - len(filtered)))
    return filtered

#filters out categories with less than cat_threshold documents (and filters documents containing those categories)
def _filter_by_frequency(doclist, cat_threshold):
    cat_count = Counter()
    for d in doclist:
        cat_count.update(d.categories)

    freq_categories = [cat for cat,count in cat_count.items() if count>cat_threshold]
    freq_categories.sort()
    return _filter_by_category(doclist, freq_categories), freq_categories

#select top most_frequent categories (and filters documents containing those categories)
def _most_common(doclist, most_frequent):
    cat_count = Counter()
    for d in doclist:
        cat_count.update(d.categories)

    freq_categories = [cat for cat,count in cat_count.most_common(most_frequent)]
    freq_categories.sort()
    return _filter_by_category(doclist, freq_categories), freq_categories

def _get_categories(request):
    final_cats = set()
    for d in request:
        final_cats.update(d.categories)
    return list(final_cats)

def fetch_jrcacquis(lang='en', data_path=None, years=None, ignore_unclassified=True,
                    cat_filter=None, cat_threshold=0, most_frequent=-1,
                    DOWNLOAD_URL_BASE ='http://optima.jrc.it/Acquis/JRC-Acquis.3.0/corpus/'):

    if not data_path:
        data_path = get_data_home()

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    request = []
    total_read = 0
    file_name = 'jrc-' + lang + '.tgz'
    archive_path = join(data_path, file_name)

    if not os.path.exists(archive_path):
        print("downloading language-specific dataset (once and for all) into %s" % data_path)
        DOWNLOAD_URL = join(DOWNLOAD_URL_BASE, file_name)
        download_file(DOWNLOAD_URL, archive_path)
        print("untarring dataset...")
        tarfile.open(archive_path, 'r:gz').extractall(data_path)

    documents_dir = join(data_path, lang)

    print("Reading documents...")
    read = 0
    for dir in list_dirs(documents_dir):
        year = int(dir)
        if years==None or year in years:
            year_dir = join(documents_dir,dir)
            l_y_documents = []
            all_documents = list_files(year_dir)
            empty = 0
            pbar = tqdm(enumerate(all_documents))
            for i,doc_file in pbar:
                try:
                    jrc_doc = parse_document(join(year_dir, doc_file), year)
                except ValueError:
                    jrc_doc = None

                if jrc_doc and (not ignore_unclassified or jrc_doc.categories):
                    l_y_documents.append(jrc_doc)
                else: empty += 1
                read+=1
                pbar.set_description(f'from {year_dir}: discarded {empty} without categories or empty fields')
            request += l_y_documents
    print("Read %d documents for language %s\n" % (read, lang))
    total_read += read

    final_cats = _get_categories(request)

    if cat_filter:
        request = _filter_by_category(request, cat_filter)
        final_cats = _get_categories(request)
    if cat_threshold > 0:
        request, final_cats = _filter_by_frequency(request, cat_threshold)
    if most_frequent != -1 and len(final_cats) > most_frequent:
        request, final_cats = _most_common(request, most_frequent)

    return request, final_cats

def print_cat_analysis(request):
    cat_count = Counter()
    for d in request:
        cat_count.update(d.categories)
    print("Number of active categories: {}".format(len(cat_count)))
    print(cat_count.most_common())

# inspects the Eurovoc thesaurus in order to select a subset of categories
# currently, only 'broadest' policy (i.e., take all categories with no parent category), and 'all' is implemented
def inspect_eurovoc(data_path, eurovoc_skos_core_concepts_filename='eurovoc_in_skos_core_concepts.rdf',
                    eurovoc_url="http://publications.europa.eu/mdr/resource/thesaurus/eurovoc-20160630-0/skos/eurovoc_in_skos_core_concepts.zip",
                    select="broadest"):

    fullpath_pickle = join(data_path, select+'_concepts.pickle')
    if os.path.exists(fullpath_pickle):
        print("Pickled object found in %s. Loading it." % fullpath_pickle)
        return pickle.load(open(fullpath_pickle,'rb'))

    fullpath = join(data_path, eurovoc_skos_core_concepts_filename)
    if not os.path.exists(fullpath):
        print("Path %s does not exist. Trying to download the skos EuroVoc file from %s" % (data_path, eurovoc_url))
        download_file(eurovoc_url, fullpath)
        print("Unzipping file...")
        zipped = zipfile.ZipFile(data_path + '.zip', 'r')
        zipped.extract("eurovoc_in_skos_core_concepts.rdf", data_path)
        zipped.close()

    print("Parsing %s" %fullpath)
    g = rdflib.Graph()
    g.parse(location=fullpath, format="application/rdf+xml")

    if select == "all":
        print("Selecting all concepts")
        all_concepts = list(g.subjects(RDF.type, SKOS.Concept))
        all_concepts = [c.toPython().split('/')[-1] for c in all_concepts]
        all_concepts.sort()
        selected_concepts = all_concepts
    elif select=="broadest":
        print("Selecting broadest concepts (those without any other broader concept linked to it)")
        all_concepts = set(g.subjects(RDF.type, SKOS.Concept))
        narrower_concepts = set(g.subjects(SKOS.broader, None))
        broadest_concepts = [c.toPython().split('/')[-1] for c in (all_concepts - narrower_concepts)]
        broadest_concepts.sort()
        selected_concepts = broadest_concepts
    elif select=="leaves":
        print("Selecting leaves concepts (those not linked as broader of any other concept)")
        all_concepts = set(g.subjects(RDF.type, SKOS.Concept))
        broad_concepts = set(g.objects(None, SKOS.broader))
        leave_concepts = [c.toPython().split('/')[-1] for c in (all_concepts - broad_concepts)]
        leave_concepts.sort()
        selected_concepts = leave_concepts
    else:
        raise ValueError("Selection policy %s is not currently supported" % select)

    print("%d %s concepts found" % (len(selected_concepts), leave_concepts))
    print("Pickling concept list for faster further requests in %s" % fullpath_pickle)
    pickle.dump(selected_concepts, open(fullpath_pickle, 'wb'), pickle.HIGHEST_PROTOCOL)

    return selected_concepts



if __name__ == '__main__':

    train_years = list(range(1986, 2006))
    test_years = [2006]
    cat_policy = 'all'#'leaves'
    most_common_cat = 300
    JRC_DATAPATH = "../datasets/JRC_Acquis_v3"
    cat_list = inspect_eurovoc(JRC_DATAPATH, select=cat_policy)

    training_docs, tr_cats = fetch_jrcacquis(lang='en', data_path=JRC_DATAPATH, years=train_years,
                                                 cat_filter=None, cat_threshold=1,
                                                 most_frequent=most_common_cat)
    test_docs, te_cats = fetch_jrcacquis(lang='en', data_path=JRC_DATAPATH, years=test_years,
                                                 cat_filter=tr_cats, cat_threshold=1)
    # training_cats = jrc_get_categories(training_docs)
    # test_cats     = jrc_get_categories(test_docs)
    # intersection_cats = [c for c in training_cats if c in test_cats]

    # training_docs = jrc_filter_by_category(training_docs, intersection_cats)
    # test_docs = jrc_filter_by_category(test_docs, intersection_cats)


    print(f'JRC-train: {len(training_docs)} documents')
    print(f'JRC-test: {len(test_docs)} documents')

    print_cat_analysis(training_docs)
    print_cat_analysis(test_docs)

    """
JRC-train: 12615 documents, 300 cats
JRC-test: 7055 documents, 300 cats
    """


