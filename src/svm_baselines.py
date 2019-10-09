import warnings
from embedding.supervised import supervised_embeddings_tfidf
from util.multilabelsvm import MLSVC
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from util.metrics import evaluation
from data.dataset import *
from util.csv_log import CSVLog
from time import time
from scipy.sparse import issparse, csr_matrix


def cls_performance(Xtr, ytr, Xte, yte, classification_type, optimizeC=True, estimator=LinearSVC):
    print('training learner...')
    tinit = time()
    param_grid = {'C': np.logspace(-3, 3, 7)} if optimizeC else None
    cv=5
    if classification_type == 'multilabel':
        cls = MLSVC(n_jobs=-1, estimator=estimator, class_weight='balanced')
        cls.fit(Xtr, _todense(ytr), param_grid=param_grid, cv=cv)
        yte_ = cls.predict(Xte)
        Mf1, mf1, acc = evaluation(_tosparse(yte), _tosparse(yte_), classification_type)
    else:
        cls = estimator(class_weight='balanced')
        cls = GridSearchCV(cls, param_grid, cv=cv, n_jobs=-1) if optimizeC else cls
        cls.fit(Xtr, ytr)
        yte_ = cls.predict(Xte)
        Mf1, mf1, acc = evaluation(yte, yte_, classification_type)
    tend = time() - tinit
    return Mf1, mf1, acc, tend


def main():
    logfile = CSVLog(args.log_file, ['dataset', 'method', 'measure', 'value', 'timelapse'], autoflush=True)
    logfile.set_default('dataset', args.dataset)

    dataset = Dataset.load(dataset_name=args.dataset, pickle_path=args.pickle_path)

    Xtr, Xte = dataset.vectorize()
    ytr, yte = dataset.devel_target, dataset.test_target
    learner = LinearSVC if args.learner=='svm' else LogisticRegression
    learner_name = 'SVM' if args.learner=='svm' else 'LR'

    if args.mode=='tfidf':
        logfile.set_default('method', f'{learner_name}-tfidf-{"opC" if args.optimc else "default"}')
        assert not logfile.already_calculated(), f'baselines for {args.dataset} already calculated'
        Mf1, mf1, acc, tend = cls_performance(Xtr, ytr, Xte, yte, dataset.classification_type, args.optimc, learner)
        logfile.add_row(measure='te-macro-F1', value=Mf1, timelapse=tend)
        logfile.add_row(measure='te-micro-F1', value=mf1, timelapse=tend)
        logfile.add_row(measure='te-accuracy', value=acc, timelapse=tend)

    else:
        logfile.set_default('method', f'{learner_name}-S-{"opC" if args.optimc else "default"}')
        assert not logfile.already_calculated(), f'baselines for {args.dataset} already calculated'
        tinit = time()
        Y = dataset.devel_labelmatrix
        F = supervised_embeddings_tfidf(Xtr, Y)
        XFtr = Xtr.dot(F)
        XFte = Xte.dot(F)
        sup_tend = time()-tinit
        Mf1, mf1, acc, tend = cls_performance(XFtr, ytr, XFte, yte, dataset.classification_type, args.optimc, learner)
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
    parser.add_argument('--dataset', type=str, default='rcv1', metavar='N', help=f'dataset, one in {Dataset.dataset_available}')
    parser.add_argument('--pickle-path', type=str, default=None, metavar='N', help=f'if set, specifies the path where to'
                             f'save/load the dataset pickled')
    parser.add_argument('--log-file', type=str, default='../log/log.csv', metavar='N', help='path to the log csv file')
    parser.add_argument('--learner', type=str, default='svm', metavar='N', help=f'learner (svm or lr)')
    parser.add_argument('--mode', type=str, default='tfidf', metavar='N', help=f'mode, in tfidf or supervised')
    parser.add_argument('--optimc', action='store_true', default=False, help='optimize the C parameter in the SVM')
    parser.add_argument('--max-docs', type=int, default=25000, metavar='N',
                        help='maximum number of training documents to submit to the classifier. If the actual number of '
                             'training examples is higher, then at balanced undersampling is performed individually for '
                             'each category (default 25000)')
    args = parser.parse_args()

    assert args.mode in ['tfidf', 'supervised'], 'unknown mode'
    assert args.learner in ['svm', 'lr'], 'unknown learner'
    main()