#https://www.wipo.int/classifications/ipc/en/ITsupport/Categorization/dataset/
import os, sys
from os.path import exists, join
from util.file import *
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
import pickle
from joblib import Parallel, delayed

WIPO_URL= 'https://www.wipo.int/classifications/ipc/en/ITsupport/Categorization/dataset/'

class WipoGammaDocument:
    def __init__(self, id, text, main_label, all_labels):
        self.id = id
        self.text = text
        self.main_label = main_label
        self.all_labels = all_labels


def remove_nested_claimtext_tags(xmlcontent):
    from_pos = xmlcontent.find(b'<claims')
    to_pos = xmlcontent.find(b'</claims>')
    if from_pos > -1 and to_pos > -1:
        in_between = xmlcontent[from_pos:to_pos].replace(b'<claim-text>',b'').replace(b'</claim-text>',b'')
        xmlcontent = (xmlcontent[:from_pos]+in_between+xmlcontent[to_pos:]).strip()
    return xmlcontent


def parse_document(xml_content, text_fields, limit_description):
    root = ET.fromstring(remove_nested_claimtext_tags(xml_content))

    doc_id  = root.attrib['ucid']
    lang    = root.attrib['lang']

    #take categories from the categorization up the "sub-class" level
    main_group = set(t.text[:6] for t in root.findall('.//bibliographic-data/technical-data/classifications-ipcr/classification-ipcr[@computed="from_ecla_to_ipc_SG"][@generated_main_IPC="true"]'))
    sec_groups = set(t.text[:6] for t in root.findall('.//bibliographic-data/technical-data/classifications-ipcr/classification-ipcr[@computed="from_ecla_to_ipc_SG"][@generated_main_IPC="false"]'))
    sec_groups.update(main_group)

    assert len(main_group) == 1, 'more than one main groups'
    main_group = list(main_group)[0]
    sec_groups = sorted(list(sec_groups))

    # main_group = sorted(list(set(t.text[:6] for t in root.findall('.//bibliographic-data/technical-data/classifications-ipcr/classification-ipcr[@computed="from_ecla_to_ipc_MG"][@generated_main_IPC="true"]'))))
    # sec_groups = sorted(list(set(t.text[:6] for t in root.findall('.//bibliographic-data/technical-data/classifications-ipcr/classification-ipcr[@computed="from_ecla_to_ipc_MG"][@generated_main_IPC="false"]'))))

    # assert len(main_cats) == 1, 'more than one main class'

    assert lang == 'EN', f'only English documents allowed (doc {doc_id})'

    doc_text_fields=[]
    if 'abstract' in text_fields:
        abstract = '\n'.join(filter(None, [t.text for t in root.findall('.//abstract[@lang="EN"]/p')]))
        doc_text_fields.append(abstract)
    if 'description' in text_fields:
        description = '\n'.join(filter(None, [t.text for t in root.findall('.//description[@lang="EN"]/p')]))
        if limit_description>-1:
            description=' '.join(description.split()[:limit_description])
        doc_text_fields.append(description)
    if 'claims' in text_fields:
        claims = '\n'.join(filter(None, [t.text for t in root.findall('.//claims[@lang="EN"]/claim')]))
        doc_text_fields.append(claims)

    text = '\n'.join(doc_text_fields)
    if text:
        return WipoGammaDocument(doc_id, text, main_group, sec_groups)
    else:
        return None


def extract(fin, fout, text_fields, limit_description):
    zipfile = ZipFile(fin)
    ndocs=0
    with open(fout, 'wt') as out:
        for xmlfile in tqdm(zipfile.namelist()):
            if xmlfile.endswith('.xml'):
                xmlcontent = zipfile.open(xmlfile).read()
                document = parse_document(xmlcontent, text_fields, limit_description)
                if document:
                    line_text = document.text.replace('\n', ' ').replace('\t', ' ')
                    line_secgroups = ' '.join(document.secondary_groups)
                    out.write('\t'.join([document.id, document.main_group, line_secgroups, line_text]))
                    out.write('\n')
                    ndocs+=1
            out.flush()


# wipo_pickle = 'wipo-gamma.pickle'
# if not exists(wipo_pickle):
# print(f'extracting {text_fields} from WIPO-gamma ({limit_description} words from description) [will take some minutes...]')
# Parallel(n_jobs=-1)(delayed(extract)(join(data_path,file), join(data_path_out,file.replace('.zip','.txt'))) for file in list_files(data_path))

    # pickle.dump(documents, open(wipo_pickle, 'wb'), pickle.HIGHEST_PROTOCOL)
# else:
#     print(f'loading {wipo_pickle}')
#     with open(wipo_pickle, 'rb') as fo:
#         documents = pickle.load(fo)

def read_classification_file(data_path, classification_level):
    assert classification_level in ['subclass', 'maingroup'], 'wrong classification requested'
    z = ZipFile(join(data_path,'EnglishWipoGamma1.zip'))
    inpath='Wipo_Gamma/English/TrainTestSpits'
    document_labels = dict()
    train_ids, test_ids = set(), set()
    labelcut = LabelCut(classification_level)
    for subset in tqdm(['train', 'test'], desc='loading classification file'):
        target_subset = train_ids if subset=='train' else test_ids
        if classification_level == 'subclass':
            file = f'{subset}set_en_sc.parts' #sub-class level
        else:
            file = f'{subset}set_en_mg.parts' #main-group level

        for line in z.open(f'{inpath}/{file}').readlines():
            line = line.decode().strip().split(',')
            id = line[0]
            id = id[id.rfind('/')+1:].replace('.xml','')
            labels = labelcut.trim(line[1:])
            document_labels[id]=labels
            target_subset.add(id)

    return document_labels, train_ids, test_ids

class LabelCut:
    """
    Labels consists of 1 char for section, 2 chars for class, 1 class for subclass, 2 chars for maingroup and so on.
    This class cuts the label at a desired level (4 for subclass, or 6 for maingroup)
    """
    def __init__(self, classification_level):
        assert classification_level in {'subclass','maingroup'}, 'unknown classification level'
        if classification_level == 'subclass': self.cut = 4
        else: self.cut = 6

    def trim(self, label):
        if isinstance(label, list):
            return sorted(set([l[:self.cut] for l in label]))
        else:
            return label[:self.cut]



def fetch_WIPOgamma(subset, classification_level, data_home, extracted_path, text_fields = ('abstract', 'description'), limit_description=300):
    """
    Fetchs the WIPO-gamma dataset
    :param subset: 'train' or 'test' split
    :param classification_level: the classification level, either 'subclass' or 'maingroup'
    :param data_home: directory containing the original 11 English zips
    :param extracted_path: directory used to extract and process the original files
    :param text_fields: indicates the fields to extract, in 'abstract', 'description', 'claims'
    :param limit_description: the maximum number of words to take from the description field (default 300); set to -1 for all
    :return:
    """
    assert subset in {"train", "test"}, 'unknown target request (valid ones are "train" or "test")'
    assert len(text_fields)>0, 'at least some text field should be indicated'
    if not exists(data_home):
        raise ValueError(f'{data_home} does not exist, and the dataset cannot be automatically download, '
              f'since you need to request for permission. Please refer to {WIPO_URL}')

    create_if_not_exist(extracted_path)
    config = f'{"-".join(text_fields)}-{limit_description}'
    pickle_path=join(extracted_path, f'wipo-{subset}-{classification_level}-{config}.pickle')
    if exists(pickle_path):
        print(f'loading pickled file in {pickle_path}')
        return pickle.load(open(pickle_path,'rb'))

    print('pickle file not found, processing...(this will take some minutes)')
    extracted = sum([exists(f'{extracted_path}/EnglishWipoGamma{(i+1)}-{config}.txt') for i in range(11)])==11
    if not extracted:
        print(f'extraction files not found, extracting files in {data_home}... (this will take some additional minutes)')
        Parallel(n_jobs=-1)(
            delayed(extract)(
                join(data_home, file), join(extracted_path, file.replace('.zip', f'-{config}.txt')), text_fields, limit_description
            )
            for file in list_files(data_home)
        )
    doc_labels, train_ids, test_ids = read_classification_file(data_home, classification_level=classification_level)  # or maingroup
    print(f'{len(doc_labels)} documents classified split in {len(train_ids)} train and {len(test_ids)} test documents')

    train_request = []
    test_request  = []
    pbar = tqdm(list_files(extracted_path))
    labelcut = LabelCut(classification_level)
    errors=0
    for proc_file in pbar:
        pbar.set_description(f'processing {proc_file} [errors={errors}]')
        if not proc_file.endswith(f'-{config}.txt'): continue
        lines = open(f'{extracted_path}/{proc_file}', 'rt').readlines()
        for lineno,line in enumerate(lines):
            parts = line.split('\t')
            assert len(parts)==4, f'wrong format in {extracted_path}/{proc_file} line {lineno}'
            id,mainlabel,alllabels,text=parts
            mainlabel = labelcut.trim(mainlabel)
            alllabels = labelcut.trim(alllabels.split())

            # assert id in train_ids or id in test_ids, f'id {id} out of scope'
            if id not in train_ids and id not in test_ids:
                errors+=1
            else:
                # assert mainlabel == doc_labels[id][0], 'main label not consistent'
                request = train_request if id in train_ids else test_request
                request.append(WipoGammaDocument(id, text, mainlabel, alllabels))

    print('pickling requests for faster subsequent runs')
    pickle.dump(train_request, open(join(extracted_path,f'wipo-train-{classification_level}-{config}.pickle'), 'wb', pickle.HIGHEST_PROTOCOL))
    pickle.dump(test_request, open(join(extracted_path, f'wipo-test-{classification_level}-{config}.pickle'), 'wb', pickle.HIGHEST_PROTOCOL))

    if subset== 'train':
        return train_request
    else:
        return test_request


if __name__=='__main__':
    data_home = '../../datasets/WIPO/wipo-gamma/en'
    extracted_path = '../../datasets/WIPO-extracted'

    train = fetch_WIPOgamma(subset='train', classification_level='subclass', data_home=data_home, extracted_path=extracted_path)
    test = fetch_WIPOgamma(subset='test', classification_level='subclass', data_home=data_home, extracted_path=extracted_path)
    train = fetch_WIPOgamma(subset='train', classification_level='maingroup', data_home=data_home, extracted_path=extracted_path)
    test = fetch_WIPOgamma(subset='test', classification_level='maingroup', data_home=data_home, extracted_path=extracted_path)

    print('Done')
