import warnings

from model.CustomRepresentationLearning import CustomRepresentationModel
from util import file

warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.feature_extraction.text import CountVectorizer
from supervised_term_weighting.supervised_vectorizer import TSRweighting
from supervised_term_weighting.tsr_functions import *
from util.multilabelsvm import MLSVC
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from util.metrics import evaluation
from data.dataset import *
from util.csv_log import CSVLog
from time import time
from scipy.sparse import issparse, csr_matrix
from embedding.pretrained import GloVe
from embedding.supervised import get_supervised_embeddings


def cls_performance(Xtr, ytr, Xte, yte, classification_type, optimizeC=True, estimator=LinearSVC,
                    class_weight='balanced'):
    print('training learner...')
    tinit = time()
    param_grid = {'C': np.logspace(-3, 3, 7)} if optimizeC else None
    cv = 5
    if classification_type == 'multilabel':
        cls = MLSVC(n_jobs=-1, estimator=estimator, class_weight=class_weight)
        cls.fit(Xtr, _todense(ytr), param_grid=param_grid, cv=cv)
        yte_ = cls.predict(Xte)
        Mf1, mf1, acc = evaluation(_tosparse(yte), _tosparse(yte_), classification_type)
    else:
        cls = estimator(class_weight=class_weight)
        cls = GridSearchCV(cls, param_grid, cv=cv, n_jobs=-1) if optimizeC else cls
        cls.fit(Xtr, ytr)
        yte_ = cls.predict(Xte)
        Mf1, mf1, acc = evaluation(yte, yte_, classification_type)
    tend = time() - tinit
    return Mf1, mf1, acc, tend


def embedding_matrix(dataset, pretrained=False, supervised=False):
    assert pretrained or supervised, 'useless call without requiring pretrained and/or supervised embeddings'
    vocabulary = dataset.vocabulary
    vocabulary = np.asarray(list(zip(*sorted(vocabulary.items(), key=lambda x: x[1])))[0])

    print('[embedding matrix]')
    pretrained_embeddings = []
    if pretrained:
        print('\t[pretrained-matrix: GloVe]')
        pretrained = GloVe()
        P = pretrained.extract(vocabulary).numpy()
        pretrained_embeddings.append(P)
        del pretrained

    if supervised:
        print('\t[supervised-matrix]')
        Xtr, _ = dataset.vectorize()
        Ytr = dataset.devel_labelmatrix
        S = get_supervised_embeddings(Xtr, Ytr)
        pretrained_embeddings.append(S)

    pretrained_embeddings = np.hstack(pretrained_embeddings)
    print(f'[embedding matrix done] of shape={pretrained_embeddings.shape}')

    return vocabulary, pretrained_embeddings


def tsr(name):
    name = name.lower()
    if name == 'ig':
        return information_gain
    elif name == 'pmi':
        return pointwise_mutual_information
    elif name == 'gr':
        return gain_ratio
    elif name == 'chi':
        return chi_square
    elif name == 'rf':
        return relevance_frequency
    elif name == 'cw':
        return conf_weight
    else:
        raise ValueError(f'unknown function {name}')


def main(args):
    logfile = CSVLog(args.log_file, ['dataset', 'method', 'measure', 'value', 'timelapse'], autoflush=True)
    logfile.set_default('dataset', args.dataset)
    learner = LinearSVC if args.learner == 'svm' else LogisticRegression
    learner_name = 'SVM' if args.learner == 'svm' else 'LR'
    mode = args.mode
    if args.mode == 'stw':
        mode += f'-{args.tsr}-{args.stwmode}'
    method_name = f'{learner_name}-{mode}-{"opC" if args.optimc else "default"}'

    logfile.set_default('method', method_name)
    assert not logfile.already_calculated() or args.force, f'baseline {method_name} for {args.dataset} already calculated'

    dataset = Dataset.load(dataset_name=args.dataset,
                           pickle_path=os.path.join(args.pickle_dir, f'{args.dataset}.pickle'))

    class_weight = 'balanced' if args.balanced else None
    print(f'running with class_weight={class_weight}')

    # tfidf = TfidfVectorizer(min_df=5)
    # Xtr = tfidf.fit_transform(dataset.devel_raw)
    # Xte = tfidf.transform(dataset.test_raw)
    ytr, yte = dataset.devel_target, dataset.test_target
    if args.mode == 'stw':
        print('Supervised Term Weighting')
        coocurrence = CountVectorizer(vocabulary=dataset.vocabulary)
        Ctr = coocurrence.transform(dataset.devel_raw)
        Cte = coocurrence.transform(dataset.test_raw)
        stw = TSRweighting(tsr_function=tsr(args.tsr), global_policy=args.stwmode)
        Xtr = stw.fit_transform(Ctr, dataset.devel_labelmatrix)
        Xte = stw.transform(Cte)
    else:
        Xtr, Xte = dataset.vectorize()

    if mode in ['bert', 'bert-sup']:
        # load best model and get document embeddings for the dataset
        bert_filename = os.path.join(args.pickle_dir, f'{args.dataset}_BERTembeddings_{args.combine_strategy}.pickle')
        if file.exists(bert_filename) and not args.force_embeddings:
            print('Loading pre-computed BERT document embeddings')
            with open(bert_filename, mode='rb') as inputfile:
                NLMtr, NLMte = pickle.load(inputfile)
        else:
            print('Computing BERT document embeddings')
            model = CustomRepresentationModel('bert', os.path.join(args.model_dir, args.dataset, 'best_model'))

            NLMtr = model.encode_sentences(dataset.devel_raw, combine_strategy=args.combine_strategy,
                                           batch_size=args.batch_size)
            NLMte = model.encode_sentences(dataset.test_raw, combine_strategy=args.combine_strategy,
                                           batch_size=args.batch_size)

            with open(bert_filename, mode='wb') as outputfile:
                pickle.dump((NLMtr, NLMte), outputfile)

    if args.mode in ['tfidf', 'stw', 'bert']:
        sup_tend = 0
    else:
        tinit = time()
        pretrained, supervised = False, False
        if args.mode in ['sup', 'bert-sup']:
            supervised = True
        elif args.mode == 'glove':
            pretrained = True
        elif args.mode == 'glove-sup':
            pretrained, supervised = True, True
        _, F = embedding_matrix(dataset, pretrained=pretrained, supervised=supervised)
        Xtr = Xtr.dot(F)
        Xte = Xte.dot(F)
        sup_tend = time() - tinit

    # concatenating documents vectors from indexing with those from BERT model
    if mode == 'bert':
        Xtr = NLMtr
        Xte = NLMte
    elif mode == 'bert-sup':
        Xtr = np.hstack((Xtr, NLMtr))
        Xte = np.hstack((Xte, NLMte))

    print(Xtr.shape, Xte.shape)

    Mf1, mf1, acc, tend = cls_performance(Xtr, ytr, Xte, yte, dataset.classification_type, args.optimc, learner,
                                          class_weight=class_weight)
    tend += sup_tend
    logfile.add_row(measure='te-macro-F1', value=Mf1, timelapse=tend)
    logfile.add_row(measure='te-micro-F1', value=mf1, timelapse=tend)
    logfile.add_row(measure='te-accuracy', value=acc, timelapse=tend)

    print('Done!')


def _todense(y):
    return y.toarray() if issparse(y) else y


def _tosparse(y):
    return y if issparse(y) else csr_matrix(y)


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='Text Classification with Embeddings')
    parser.add_argument('--dataset', type=str, default='rcv1', metavar='N',
                        help=f'dataset, one in {Dataset.dataset_available}')
    parser.add_argument('--pickle-dir', type=str, default='../pickles', metavar='str',
                        help=f'path where to load the pickled dataset from and to save BERT document embeddings')
    parser.add_argument('--log-file', type=str, default='../log/log.csv', metavar='N', help='path to the log csv file')
    parser.add_argument('--learner', type=str, default='svm', metavar='N', help=f'learner (svm or lr)')
    parser.add_argument('--mode', type=str, default='tfidf', metavar='N',
                        help=f'mode, in tfidf, stw, sup, glove, glove-sup, bert, bert-sup')
    parser.add_argument('--stwmode', type=str, default='wave', metavar='N',
                        help=f'mode in which the term relevance will be merged (wave, ave, max). Only for --mode stw. '
                             f'Default "wave"')
    parser.add_argument('--tsr', type=str, default='ig', metavar='TSR',
                        help=f'indicates the accronym of the TSR function to use in supervised term weighting '
                             f'(only if --mode stw). Valid functions are '
                             f'IG (information gain), '
                             f'PMI (pointwise mutual information) '
                             f'GR (gain ratio) '
                             f'CHI (chi-square) '
                             f'RF (relevance frequency) '
                             f'CW (ConfWeight)')
    parser.add_argument('--optimc', action='store_true', default=False, help='optimize the C parameter in the SVM')
    parser.add_argument('--balanced', action='store_true', default=False, help='class weight balanced')
    parser.add_argument('--combine-strategy', default=None, type=str,
                        help='Method to determine BERT document embeddings.'
                             'No value takes the [CLS] embedding.'
                             '"mean" makes the mean of token embeddings.')
    parser.add_argument('--model-dir', type=str, default='../models', metavar='str',
                        help=f'path where the BERT model is stored. Dataset name is added')
    parser.add_argument('--force-embeddings', action='store_true', default=False,
                        help='force the computation of embeddings even if a precomputed version is available')
    parser.add_argument('--batch-size', type=int, default=512, metavar='int',
                        help='batch size for computation of BERT document embeddings')
    parser.add_argument('--force', action='store_true', default=False,
                        help='force the execution of the experiment even if a log already exists')
    args = parser.parse_args()
    assert args.mode in ['tfidf', 'sup', 'glove', 'glove-sup', 'stw', 'bert', 'bert-sup'], 'unknown mode'
    assert args.mode != 'stw' or args.tsr in ['ig', 'pmi', 'gr', 'chi', 'rf', 'cw'], 'unknown tsr'
    assert args.stwmode in ['wave', 'ave', 'max'], 'unknown stw-mode'
    assert args.learner in ['svm', 'lr'], 'unknown learner'
    assert args.combine_strategy in [None, 'mean'], 'unknown combine strategy'

    if args.combine_strategy is None:
        args.combine_strategy = 0

    main(args)
