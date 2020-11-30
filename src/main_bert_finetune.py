import argparse
import os
from time import time

import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs, MultiLabelClassificationModel
from simpletransformers.config.model_args import MultiLabelClassificationArgs
from sklearn.metrics import classification_report, confusion_matrix, f1_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

from data.dataset import Dataset


def train_val_test(dataset, seed):
    val_size = min(int(len(dataset.devel_raw) * .2), 20000)
    train_docs, val_docs, ytr, yval = train_test_split(
        dataset.devel_raw, dataset.devel_target, test_size=val_size, random_state=seed, shuffle=True
    )
    return (train_docs, ytr), (val_docs, yval), (dataset.test_raw, dataset.test_target)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Neural text classification with Word-Class Embeddings',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, metavar='str', help=f'dataset, one in {Dataset.dataset_available}')
    parser.add_argument('--max-epochs', type=int, default=50, metavar='int', help='max number of epochs')
    parser.add_argument('--patience', type=int, default=5, metavar='int', help='patience for early-stop')
    parser.add_argument('--sample-size', type=int, default=0, metavar='int', help='sample size for epoch, 0 means full training set')
    parser.add_argument('--seed', type=int, default=1, metavar='int',
                        help='random seed')
    parser.add_argument('--pickle-dir', type=str, default='../pickles', metavar='str',
                        help=f'path where to load the pickled dataset from')
    parser.add_argument('--model-dir', type=str, default='../models', metavar='str',
                        help=f'path where fitted model will be saved. Dataset name is added')

    opt = parser.parse_args()

    if not opt.dataset:
        parser.error('Missing dataset name')

    dataset_name = opt.dataset

    dataset = Dataset.load(dataset_name=dataset_name,
                           pickle_path=os.path.join(opt.pickle_dir, f'{dataset_name}.pickle'))

    dataset.show()

    singlelabel = dataset.classification_type == 'singlelabel'

    if singlelabel:
        ModelArgsClass = ClassificationArgs
        ModelClass = ClassificationModel
        confusion_function = confusion_matrix
    else:
        ModelArgsClass = MultiLabelClassificationArgs
        ModelClass = MultiLabelClassificationModel
        confusion_function = multilabel_confusion_matrix

    # dataset split tr/val/test
    (train_docs, ytr), (val_docs, yval), (test_docs, yte) = train_val_test(dataset, opt.seed)

    if singlelabel:
        train_df = pd.DataFrame(zip(train_docs, ytr))
        train_df.columns = ['text', 'labels']
    else:
        yval = yval.todense().tolist()
        yte = yte.todense().tolist()

        train_df = pd.DataFrame(zip(train_docs, ytr.todense().tolist()))
        train_df.columns = ['text', 'labels']

    # Optional model configuration
    model_args = ModelArgsClass()
    model_args.num_train_epochs = 1
    model_args.overwrite_output_dir = True
    # we do validation in an external loop
    model_args.evaluate_during_training = False
    model_args.evaluate_each_epoch = False
    model_args.do_lower_case = True
    model_args.save_best_model = False
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = False
    model_args.save_steps = 0
    model_args.output_dir = os.path.join(opt.model_dir, dataset_name)

    # Create a ClassificationModel
    model = ModelClass("bert", "bert-base-uncased", num_labels=dataset.nC, args=model_args)

    np.set_printoptions(threshold=np.inf, linewidth=10000000000000)

    # run every epoch and evaluate on validation and test set
    max_val_f1_macro = 0
    best_epoch = -1
    patience = 5
    missed = 0
    max_epochs = 50
    start = time()
    for i in range(max_epochs):
        # train
        print(f'Training {dataset_name}, epoch = {i + 1}')
        model.args.no_save = (i + 1) < max_epochs

        if opt.sample_size>0:
            epoch_df = train_df.sample(opt.sample_size)
        else:
            epoch_df = train_df

        model.train_model(epoch_df, model_args=model_args)
        print(f'Training time = {time() - start}')

        # validation
        results = model.predict(val_docs)

        print(f'Validation {dataset_name}, epoch = {i + 1}')
        print('Validation classification report:')
        print(classification_report(yval, results[0], digits=3))
        print('Validation confusion matrix:')
        cm = confusion_function(yval, results[0])
        print(cm)
        val_f1_macro = f1_score(yval, results[0], average='macro')
        if val_f1_macro > max_val_f1_macro and val_f1_macro > 0:
            model.args.no_save = False
            model.save_model(model=model.model, output_dir=model_args.output_dir + '/best_model')
            missed = 0
            max_val_f1_macro = val_f1_macro
            best_epoch = i
        else:
            missed += 1
        print(f'Validation time = {time() - start}')
        print(f'max_val_f1={max_val_f1_macro}, missed={missed}/{patience}')
        if missed == patience:
            break

    # load best model and test
    model = ModelClass('bert', model_args.output_dir + '/best_model', num_labels=dataset.nC, args=model_args)
    results = model.predict(test_docs)

    print(f'Test {dataset_name}, epoch = {best_epoch + 1}')
    print('Test classification report:')
    print(classification_report(yte, results[0], digits=3))
    print('Test confusion matrix:')
    cm = confusion_function(yte, results[0])
    print(cm)
    print(f'Test time = {time() - start}')
