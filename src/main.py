import argparse
import torch
import torchtext
import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import scipy
from baselines import svm_performance
from embedding.supervised import get_supervised_embeddings, fit_predict, multi_domain_sentiment_embeddings
from model.cnn_class import CNN
from model.lstm_attn_class import AttentionModel
from model.lstm_class import LSTMClassifier
from util.early_stop import EarlyStopping
from common import *
from data.dataset import *
from util.csv_log import CSVLog
from util.file import create_if_not_exist
from util.metrics import *
from time import time
from embedding.pretrained import *

#TODO: other embeddings (unsupervised: FASTTEXT, SentiWordNet?; supervised: ?)
#TODO: control vocabulary (max-size 200k? drop GloVe? min_df=1?)

allowed_nets = {'cnn', 'lstm', 'attn'}

def init_Net(nC, vocabsize, pretrained_embeddings, sup_range, xavier_uniform=True, tocuda=True):
    net=opt.net
    assert net in allowed_nets, f'{net} not supported, valid ones are={allowed_nets}'

    if net=='lstm':
        model = LSTMClassifier(output_size=nC, hidden_size=opt.hidden, vocab_size=vocabsize,
                               learnable_length=opt.learnable, pretrained=pretrained_embeddings,
                               drop_embedding_range=sup_range, drop_embedding_prop=opt.sup_drop)
    elif net=='attn':
        model = AttentionModel(output_size=nC, hidden_size=opt.hidden, vocab_size=vocabsize,
                               learnable_length=opt.learnable, pretrained=pretrained_embeddings,
                               drop_embedding_range=sup_range, drop_embedding_prop=opt.sup_drop)
    elif net == 'cnn':
        model = CNN(output_size=nC, out_channels=opt.channels, vocab_size=vocabsize,
                    learnable_length=opt.learnable,
                    pretrained=pretrained_embeddings, #unk_id=unk_index, pad_id=pad_index, word_drop=0,
                    drop_embedding_range=sup_range, drop_embedding_prop=opt.sup_drop)

    if xavier_uniform:
        for p in model.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)

    if tocuda:
        model=model.cuda()

    return model

def set_method_name():
    method_name = opt.net
    if opt.pretrained:
        method_name += f'-{opt.pretrained}'
    if opt.learnable > 0:
        method_name += f'-learn{opt.learnable}'
    if opt.supervised:
        method_name += '-supervised'
        if (opt.predict_missing or opt.predict_all) and opt.pretrained:
            if opt.predict_all:
                method_name+='-all'
            elif opt.predict_missing:
                method_name += '-miss'
        # if opt.sup_drop == 0:
        method_name += f'-d{opt.sup_drop}'

        method_name += f'-{opt.supervised_method}'
        if opt.supervised_nozscore:
            method_name += '-noZ'
    if (opt.pretrained or opt.supervised) and opt.tunable:
        method_name+='-tunable'
    if opt.sentiment:
        method_name += '-sent'
    if opt.weight_decay > 0:
        method_name+=f'_wd{opt.weight_decay}'
    if opt.net in {'lstm', 'attn'}:
        method_name+=f'-h{opt.hidden}'
    if opt.net== 'cnn' and opt.channels!=64:
        method_name+=f'-ch{opt.channels}'

    return method_name


def index_dataset(dataset, pretrained=None):
    # build vocabulary
    word2index = dict(dataset.vocabulary)
    known_words = set(word2index.keys())
    if pretrained is not None:
        known_words.update(pretrained.vocabulary())

    word2index['UNKTOKEN'] = len(word2index)
    word2index['PADTOKEN'] = len(word2index)
    unk_index = word2index['UNKTOKEN']
    pad_index = word2index['PADTOKEN']

    # index documents and keep track of test terms outside the development vocabulary that are in GloVe (if available)
    out_of_vocabulary = dict()
    analyzer = dataset.analyzer()
    devel_index = index(dataset.devel_raw, word2index, known_words, analyzer, unk_index, out_of_vocabulary)
    test_index = index(dataset.test_raw, word2index, known_words, analyzer, unk_index, out_of_vocabulary)

    print('[indexing complete]')
    return word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index


def embedding_matrix(dataset, pretrained, vocabsize, word2index, out_of_vocabulary):
    print('[embedding matrix]')
    pretrained_embeddings = None
    sup_range = None
    if opt.pretrained or opt.supervised or opt.sentiment:
        print('\t[pretrained-matrix]')
        pretrained_embeddings = []
        word_list = None
        if pretrained is not None:
            word_list = get_word_list(word2index, out_of_vocabulary)
            weights = pretrained.extract(word_list)
            pretrained_embeddings.append(weights)
            del pretrained

        sup_range = None
        if opt.supervised:
            print('\t[supervised-matrix]')
            Xtr, _ = dataset.vectorize()
            Ytr = dataset.devel_labelmatrix
            F = get_supervised_embeddings(Xtr, Ytr, binary_structural_problems=-1, method=opt.supervised_method, dozscore=not opt.supervised_nozscore)

            if (opt.predict_missing or opt.predict_all) and opt.pretrained:
                F = fit_predict(weights, F, mode='all' if opt.predict_all else 'missing')
            else:
                num_missing_rows = vocabsize - F.shape[0]
                F = np.vstack((F, np.zeros(shape=(num_missing_rows, F.shape[1]))))
            F = torch.from_numpy(F).float()

            offset = 0
            if pretrained_embeddings:
                offset = pretrained_embeddings[0].shape[1]
            sup_range = [offset, offset + F.shape[1]]
            pretrained_embeddings.append(F)

        if opt.sentiment:
            print('\t[sentiment-matrix]')
            sentiment_embedding = multi_domain_sentiment_embeddings()
            if word_list is None:
                word_list = get_word_list(word2index, out_of_vocabulary)
            S = sentiment_embedding.extract(word_list)
            S = torch.from_numpy(S).float()
            pretrained_embeddings.append(S)
            if sup_range:
                sup_range[1] += S.shape[1]
                print(f'sup_range={sup_range}')

        pretrained_embeddings = torch.cat(pretrained_embeddings, dim=1)

    print('[embedding matrix done]')
    return pretrained_embeddings, sup_range

def init_optimizer(model, lr):
    return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=opt.weight_decay)

def main():
    # init the log-file
    method_name = set_method_name()
    logfile = CSVLog(opt.log_file, ['dataset', 'method', 'epoch', 'measure', 'value', 'run', 'timelapse'])
    logfile.set_default('dataset', opt.dataset)
    logfile.set_default('run', opt.seed)
    logfile.set_default('method', method_name)
    assert opt.force or not logfile.already_calculated(), f'results for dataset {opt.dataset} method {method_name} and run {opt.seed} already calculated'

    # load dataset
    pretrained=None
    if opt.pretrained== 'glove':
        pretrained = GloVe()
    elif opt.pretrained== 'word2vec':
        pretrained = Word2Vec(path=opt.word2vec_path, limit=1000000)

    dataset = Dataset.load(dataset_name=opt.dataset, pickle_path=opt.pickle_path).show()
    if opt.fullpickle and os.path.exists(opt.fullpickle):
        print(f'loaded pre-computed index from {opt.fullpickle}')
        word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = pickle.load(open(opt.fullpickle, 'rb'))
    else:
        word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(dataset, pretrained)
        if opt.fullpickle:
            print(f'dumping a full pickle in {opt.fullpickle}')
            pickle.dump((word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index), open(opt.fullpickle, 'wb'), pickle.HIGHEST_PROTOCOL)
            print('[Done]')

    # dataset split tr/val/test
    val_size = min(int(len(devel_index) * .2), 20000)
    train_index, val_index, ytr, yval = train_test_split(devel_index, dataset.devel_target, test_size=val_size, random_state=opt.seed, shuffle=True)
    yte = dataset.test_target
    print(ytr.mean())

    vocabsize = len(word2index) + len(out_of_vocabulary)
    pretrained_embeddings, sup_range = embedding_matrix(dataset, pretrained, vocabsize, word2index, out_of_vocabulary)

    # instantiate the net
    model = init_Net(dataset.nC, vocabsize, pretrained_embeddings, sup_range, tocuda=True)
    if opt.tunable:
        model.finetune_pretrained()

    # init the optimizer and criterion
    optim = init_optimizer(model, lr=opt.lr)
    #TODO: scheduler?

    if dataset.classification_type == 'multilabel':
        criterion = torch.nn.BCEWithLogitsLoss().cuda()
    elif dataset.classification_type == 'singlelabel':
        criterion = torch.nn.CrossEntropyLoss().cuda()

    # train-validate
    tinit = time()
    create_if_not_exist(opt.checkpoint_dir)
    early_stop = EarlyStopping(model, patience=opt.patience, checkpoint=f'{opt.checkpoint_dir}/{opt.net}-{opt.dataset}')


    for epoch in range(1, opt.nepochs + 1):
        train(model, train_index, ytr, pad_index, tinit, logfile, criterion, optim, epoch, method_name)

        # validation
        macrof1 = test(model, val_index, yval, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'va')
        early_stop(macrof1, epoch)
        if opt.test_each>0 and epoch%opt.test_each==0 and epoch<opt.nepochs:
            test(model, test_index, yte, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'te')

        if early_stop.STOP:
            print('[early-stop]')
            if opt.finetune:
                model = early_stop.restore_checkpoint()
                nparams_before = count_parameters(model)
                model.finetune_pretrained()
                nparams_after = count_parameters(model)
                optim = init_optimizer(model, lr=opt.lr/2)
                early_stop.reinit_counter()
                opt.finetune=False
                opt.finetuning=True
                print(f'[fine-tune] {nparams_before} -> {nparams_after} trainable parameters')
                logfile.add_row(epoch=epoch, measure=f'fine-tuning', value=0, timelapse=time()-tinit)
            elif not opt.plotmode:
                break

    # restores the best model according to the Mf1 of the validation set (only when plotmode==False)
    stoptime = early_stop.stop_time - tinit
    stopepoch = early_stop.best_epoch
    logfile.add_row(epoch=stopepoch, measure=f'early-stop', value=early_stop.best_score, timelapse=stoptime)

    if opt.plotmode==False:
        if opt.val_epochs>0:
            print(f'last {opt.val_epochs} epochs on the validation set')
            for val_epoch in range(1, opt.val_epochs + 1):
                train(model, val_index, yval, pad_index, tinit, logfile, criterion, optim, epoch+val_epoch, method_name)

        print('performing final evaluation')
        model = early_stop.restore_checkpoint()

        # test
        print('Training complete: testing')
        test(model, test_index, yte, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'final-te')

        if opt.svm:
            print('test with SVM')
            ytr = dataset.devel_target
            Xtfidf_tr, Xtfidf_te = dataset.vectorize()
            test_svm(model, devel_index, Xtfidf_tr, ytr, test_index, Xtfidf_te, yte, pad_index, dataset.classification_type,
                     stoptime, stopepoch, logfile, method_name)


def train(model, train_index, ytr, pad_index, tinit, logfile, criterion, optim, epoch, method_name):
    as_long = isinstance(criterion, torch.nn.CrossEntropyLoss)
    loss_history = []
    if opt.max_epoch_length is not None:
        tr_len = len(train_index)
        train_for = opt.max_epoch_length*opt.batch_size
        from_,to_= (epoch*train_for) % tr_len, ((epoch+1)*train_for) % tr_len
        print(f'epoch from {from_} to {to_}')
        if to_ < from_:
            train_index = train_index[from_:] + train_index[:to_]
            if issparse(ytr):
                ytr = vstack((ytr[from_:], ytr[:to_]))
            else:
                ytr = np.concatenate((ytr[from_:], ytr[:to_]))
        else:
            train_index=train_index[from_:to_]
            ytr = ytr[from_:to_]
    model.train()
    for idx, (batch, target) in enumerate(batchify(train_index, ytr, opt.batch_size, pad_index, as_long)):
        optim.zero_grad()
        loss = criterion(model(batch), target)
        loss.backward()
        clip_gradient(model)
        optim.step()
        loss_history.append(loss.item())

        if idx % opt.log_interval == 0:
            interval_loss = np.mean(loss_history[-opt.log_interval:])
            print(f'{opt.dataset} {method_name}{" [finetuning]" if opt.finetuning else ""} Epoch: {epoch}, Step: {idx}, Training Loss: {interval_loss:.6f}')

    mean_loss = np.mean(interval_loss)
    logfile.add_row(epoch=epoch, measure='tr_loss', value=mean_loss, timelapse=time() - tinit)
    return mean_loss


def test(model, test_index, yte, pad_index, classification_type, tinit, epoch, logfile, criterion, measure_prefix):
    model.eval()
    predictions = []
    target_long = isinstance(criterion, torch.nn.CrossEntropyLoss)
    for batch, target in tqdm(batchify(test_index, yte, batchsize=opt.batch_size_test, pad_index=pad_index, target_long=target_long), desc='evaluation: '):
        logits = model(batch)
        loss = criterion(logits, target).item()
        prediction = csr_matrix(predict(logits, classification_type=classification_type))
        predictions.append(prediction)

    #yte_ = np.vstack(predictions)
    yte_ = scipy.sparse.vstack(predictions)
    Mf1, mf1, acc = evaluation(yte, yte_, classification_type)
    print(f'[{measure_prefix}] Macro-F1={Mf1:.3f} Micro-F1={mf1:.3f} Accuracy={acc:.3f}')
    tend = time() - tinit

    # measure_prefix='va' if is_val else 'te'
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=Mf1, timelapse=tend)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=mf1, timelapse=tend)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-loss', value=loss, timelapse=tend)

    return Mf1


def test_svm(model, devel_index, Xtfidf_tr, ytr, test_index, Xtfidf_te, yte, pad_index, classification_type, stoptime, stopepoch, logfile, method_name):
    # produces the last-layer representation for input (i.e., doc-embeddings)
    def transform(model, index, pad_index):
        model.eval()
        representations = []
        for batch in tqdm(batchify_unlabelled(index, opt.batch_size_test, pad_index), desc='transforming: '):
            doc_embeddings = model.transform(batch).detach().cpu().numpy()
            representations.append(doc_embeddings)

        X = np.vstack(representations)
        return X

    if issparse(ytr): ytr = ytr.toarray()
    if issparse(yte): yte = yte.toarray()

    Xtr = transform(model, devel_index, pad_index)
    Xte = transform(model, test_index, pad_index)
    print(f'Xtr.shape={Xtr.shape}')
    Mf1, mf1, acc, tend = svm_performance(Xtr, ytr, Xte, yte, classification_type)
    measure_prefix = 'te'
    timelapse = stoptime + tend
    logfile.set_default('method', f'SVM-{method_name}')
    logfile.add_row(epoch=stopepoch, measure=f'{measure_prefix}-macro-F1', value=Mf1, timelapse=timelapse)
    logfile.add_row(epoch=stopepoch, measure=f'{measure_prefix}-micro-F1', value=mf1, timelapse=timelapse)
    logfile.add_row(epoch=stopepoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=timelapse)

    Xtr = scipy.sparse.hstack((Xtr, Xtfidf_tr))
    Xte = scipy.sparse.hstack((Xte, Xtfidf_te))
    print(f'Xtr(concat).shape={Xtr.shape}')
    Mf1, mf1, acc, tend = svm_performance(Xtr, ytr, Xte, yte, classification_type)
    measure_prefix = 'te'
    timelapse = stoptime + tend
    logfile.set_default('method', f'SVMc-{method_name}')
    logfile.add_row(epoch=stopepoch, measure=f'{measure_prefix}-macro-F1', value=Mf1, timelapse=timelapse)
    logfile.add_row(epoch=stopepoch, measure=f'{measure_prefix}-micro-F1', value=mf1, timelapse=timelapse)
    logfile.add_row(epoch=stopepoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=timelapse)

    return Mf1


if __name__ == '__main__':
    available_datasets = Dataset.dataset_available
    available_nets = {'lstm', 'attn', 'cnn'}

    # Training settings
    parser = argparse.ArgumentParser(description='Text Classification with Embeddings')
    parser.add_argument('--dataset', type=str, default='reuters21578', metavar='N', help=f'dataset, one in {available_datasets}')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size (default: 100)')
    parser.add_argument('--batch-size-test', type=int, default=250, metavar='N', help='batch size for testing (default: 250)')
    parser.add_argument('--nepochs', type=int, default=200, metavar='N', help='number of epochs (default: 200)')
    parser.add_argument('--patience', type=int, default=5, metavar='N', help='patience for early-stop (default: 5)')
    parser.add_argument('--plotmode', action='store_true', default=False, help='in plot mode executes a long run in order to'
                                   'to generate enough data to produce trend plots (test-each should be >0, a final-te '
                                   'is not performed)')
    parser.add_argument('--hidden', type=int, default=512, metavar='N', help='hidden lstm size (default: 512)')
    parser.add_argument('--channels', type=int, default=128, metavar='N', help='number of cnn out-channels (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0, metavar='LR', help='weight decay (default: 0)')
    parser.add_argument('--sup-drop', type=float, default=0.5, metavar='LR', help='dropout probability for the supervised matrix (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before printing training status')
    parser.add_argument('--log-file', type=str, default='../log/log.csv', metavar='N', help='path to the log csv file')
    parser.add_argument('--test-each', type=int, default=0, metavar='N', help='how many epochs to wait before invoking test (default: 0, only at the end)')
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoint', metavar='N', help='path to the directory containing checkpoints')
    parser.add_argument('--net', type=str, default='lstm', metavar='N', help=f'net, one in {available_nets}')
    parser.add_argument('--pretrained', type=str, default=None, metavar='N', help='pretrained embeddings, use "glove" or "word2vec" (default None)')
    parser.add_argument('--supervised', action='store_true', default=False, help='use supervised embeddings')
    parser.add_argument('--supervised-method', type=str, default='dotn', metavar='N', help='method used to create the supervised matrix')
    parser.add_argument('--supervised-nozscore', action='store_true', default=False, help='avoid z-score all cat-dimensions')
    parser.add_argument('--sentiment', action='store_true', default=False, help='use supervised embeddings')
    parser.add_argument('--predict-missing', action='store_true', default=False, help='predict supervised embedding for out-of-vocabulary')
    parser.add_argument('--predict-all', action='store_true', default=False, help='predict supervised embedding for all words in vocabulary')
    parser.add_argument('--learnable', type=int, default=0, metavar='N', help='dimension of the learnable embeddings (default 0)')
    parser.add_argument('--val-epochs', type=int, default=1, metavar='N', help='number of training epochs to perform on the '
                        'validation set once training is over (default 1)')
    parser.add_argument('--pickle-dir', type=str, default='../pickles', metavar='N', help=f'if set, specifies the path where to'
                        f'save/load the dataset pickled (set to None if you prefer not to retain the pickle file)')
    parser.add_argument('--word2vec-path', type=str, default='../datasets/Word2Vec/GoogleNews-vectors-negative300.bin',
                        metavar='N', help=f'path to GoogleNews-vectors-negative300.bin pretrained vectors')
    parser.add_argument('--max-label-space', type=int, default=300, metavar='N', help='larger dimension allowed for the '
                        'feature-label embedding (if larger, then PCA with this number of components is applied '
                        '(default 300)')
    parser.add_argument('--max-epoch-length', type=int, default=None, metavar='N',
                        help='number of (batched) training steps before considering an epoch over (None: full epoch)') #300 for wipo-sl-sc
    parser.add_argument('--force', action='store_true', default=False, help='do not check if this experiment has already been run')
    parser.add_argument('--svm', action='store_true', default=False, help='use the learned vectors within a SVM')
    parser.add_argument('--fullpickle', type=str, default=None, metavar='N', help=f'pickles the dataset and index for fast load (None)')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='fine-tune supervised and pretrained embeddings once patience is exhausted (default False)')
    parser.add_argument('--tunable', action='store_true', default=False,
                        help='pretrained embeddings are tunable from the begining (default False)')

    # parser.add_argument('--word-drop', type=float, default=0, metavar='LR', help='word dropout probability (default: 0)')

    opt = parser.parse_args()

    assert torch.cuda.is_available(), 'CUDA not available'
    device = torch.device('cuda')
    torch.manual_seed(opt.seed)

    assert opt.dataset in available_datasets, f'unknown dataset {opt.dataset}'
    assert opt.pretrained in {None, 'glove', 'word2vec'}, f'unknown pretrained set {opt.pretrained}'
    assert not opt.plotmode or opt.test_each > 0, 'plot mode implies --test-each>0'
    assert opt.supervised_method in ['dot','pmi', 'ig', 'pnig', 'dotn', 'dotc']
    assert not (opt.tunable and opt.finetune), 'tunable and finetune cannot be activated simultaneously'

    if opt.pickle_dir: opt.pickle_path=join(opt.pickle_dir,f'{opt.dataset}.pickle')
    opt.finetuning=False

    main()
