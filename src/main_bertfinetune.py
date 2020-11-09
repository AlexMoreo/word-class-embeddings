import argparse
import math
from time import time

import scipy
import torch
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split

from data.dataset import *
from main import init_logfile, init_optimizer, init_loss
from main_bert import tokenize_and_truncate, Batcher, train, test, train_val_test
from model.classification import BertClassifier
from util.common import clip_gradient, predict, tokenize_parallel
from util.early_stop import EarlyStopping
from util.file import create_if_not_exist
from util.metrics import *


def init_Net(nC, pretrained_model_name, max_length, dropout, device):
    model = BertClassifier(
        output_size=nC,
        pretrained_model_name=pretrained_model_name,
        max_length=max_length,
        dropout=dropout,
        device=device)

    model.xavier_uniform()
    model = model.to(device)

    return model


def set_method_name():
    method_name = 'bert-finetune'
    if opt.weight_decay > 0:
        method_name += f'_wd{opt.weight_decay}'
    return method_name


def main(opt):
    method_name = set_method_name()
    logfile = init_logfile(method_name, opt)

    dataset = Dataset.load(dataset_name=opt.dataset, pickle_path=opt.pickle_path).show()
    # dataset.devel_raw=dataset.devel_raw[:100]
    # dataset.devel_target = dataset.devel_target[:100]
    # dataset.devel_labelmatrix = dataset.devel_labelmatrix[:100]

    # load model, perform tokenization
    model = init_Net(nC=dataset.nC, pretrained_model_name='bert-base-uncased', max_length=500, dropout=opt.dropout,
                     device=opt.device)
    tokenize_and_truncate(dataset, model.tokenizer, opt.max_length)

    # dataset split tr/val/test
    (train_docs, ytr), (val_docs, yval), (test_docs, yte) = train_val_test(dataset, opt.seed)

    optim = init_optimizer(model, lr=opt.lr, weight_decay=opt.weight_decay)
    criterion = init_loss(dataset.classification_type)

    # train-validate
    tinit = time()
    create_if_not_exist(opt.checkpoint_dir)
    early_stop = EarlyStopping(model, patience=opt.patience,
                               checkpoint=f'{opt.checkpoint_dir}/{opt.dataset}' if not opt.plotmode else None)

    train_batcher = Batcher(opt.batch_size, opt.max_epoch_length)
    for epoch in range(1, opt.nepochs + 1):
        train(model, train_docs, ytr, tinit, logfile, criterion, optim, epoch, method_name, train_batcher)

        # validation
        macrof1 = test(model, val_docs, yval, dataset.classification_type, tinit, epoch, logfile, criterion, 'va')
        early_stop(macrof1, epoch)
        if opt.test_each > 0:
            if (opt.plotmode and (epoch == 1 or epoch % opt.test_each == 0)) or \
                    (not opt.plotmode and epoch % opt.test_each == 0 and epoch < opt.nepochs):
                test(model, test_docs, yte, dataset.classification_type, tinit, epoch, logfile, criterion, 'te')

        if early_stop.STOP:
            print('[early-stop]')
            if not opt.plotmode:  # with plotmode activated, early-stop is ignored
                break

    # restores the best model according to the Mf1 of the validation set (only when plotmode==False)
    stoptime = early_stop.stop_time - tinit
    stopepoch = early_stop.best_epoch
    logfile.add_row(epoch=stopepoch, measure=f'early-stop', value=early_stop.best_score, timelapse=stoptime)

    if not opt.plotmode:
        print('performing final evaluation')
        model = early_stop.restore_checkpoint()

        if opt.val_epochs > 0:
            print(f'last {opt.val_epochs} epochs on the validation set')
            for val_epoch in range(1, opt.val_epochs + 1):
                train(model, val_docs, yval, tinit, logfile, criterion, optim, epoch + val_epoch, method_name)

        # test
        print('Training complete: testing')
        test(model, test_docs, yte, dataset.classification_type, tinit, epoch, logfile, criterion, 'final-te')


if __name__ == '__main__':
    available_datasets = Dataset.dataset_available

    # Training settings
    parser = argparse.ArgumentParser(description='Neural text classification using finetuning on BERT pretrained models')
    parser.add_argument('--dataset', type=str, default='reuters21578', metavar='str',
                        help=f'dataset, one in {available_datasets}')
    parser.add_argument('--batch-size', type=int, default=50, metavar='int',
                        help='input batch size (default: 100)')
    parser.add_argument('--nepochs', type=int, default=200, metavar='int',
                        help='number of epochs (default: 200)')
    parser.add_argument('--patience', type=int, default=10, metavar='int',
                        help='patience for early-stop (default: 10)')
    parser.add_argument('--plotmode', action='store_true', default=False,
                        help='in plot mode, executes a long run in order to generate enough data to produce trend plots'
                             ' (test-each should be >0. This mode is used to produce plots, and does not perform a '
                             'final evaluation on the test set other than those performed after test-each epochs).')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='float',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0, metavar='float',
                        help='weight decay (default: 0)')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='[0.0, 1.0]',
                        help='dropout probability for the classification layer (default: 0.1)')
    parser.add_argument('--seed', type=int, default=1, metavar='int',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='int',
                        help='how many batches to wait before printing training status')
    parser.add_argument('--log-file', type=str, default='../log/log.csv', metavar='str',
                        help='path to the log csv file')
    parser.add_argument('--pickle-dir', type=str, default='../pickles', metavar='str',
                        help=f'if set, specifies the path where to save/load the dataset pickled (set to None if you '
                             f'prefer not to retain the pickle file)')
    parser.add_argument('--test-each', type=int, default=0, metavar='int',
                        help='how many epochs to wait before invoking test (default: 0, only at the end)')
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoint', metavar='str',
                        help='path to the directory containing checkpoints')
    parser.add_argument('--val-epochs', type=int, default=1, metavar='int',
                        help='number of training epochs to perform on the validation set once training is '
                             'over (default 1)')
    parser.add_argument('--max-epoch-length', type=int, default=None, metavar='int',
                        help='number of (batched) training steps before considering an epoch over (None: full epoch)')  # 300 for wipo-sl-sc
    parser.add_argument('--max-length', type=int, default=250, metavar='int',
                        help='max document length (in #tokens) after which a document will be cut')
    parser.add_argument('--force', action='store_true', default=False,
                        help='do not check if this experiment has already been run')

    opt = parser.parse_args()

    opt.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'running on {opt.device}')
    torch.manual_seed(opt.seed)

    assert opt.dataset in available_datasets, f'unknown dataset {opt.dataset}'
    assert not opt.plotmode or opt.test_each > 0, 'plot mode implies --test-each>0'
    if opt.pickle_dir:
        opt.pickle_path = join(opt.pickle_dir, f'{opt.dataset}.pickle')

    main(opt)
