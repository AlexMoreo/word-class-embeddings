import argparse
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import scipy
from embedding.supervised import get_supervised_embeddings, STWFUNCTIONS
from model.classification import NeuralClassifier
from util.early_stop import EarlyStopping
from util.common import *
from data.dataset import *
from util.csv_log import CSVLog
from util.file import create_if_not_exist
from util.metrics import *
from time import time
from embedding.pretrained import *


# def init_Net(nC, vocabsize, pretrained_embeddings, sup_range, device):
#     net_type=opt.net
#     hidden = opt.channels if net_type == 'cnn' else opt.hidden
#     #if dropout for embeddings is not set, then use the supervised dropout range and prob
#     if opt.embedding_drop == 0:
#         drop_range = sup_range
#         drop_prob  = opt.sup_drop if sup_range is not None else 0
#     else:  #if otherwise, ignores the settings for supervised dropout
#         drop_range = None
#         drop_prob = opt.embedding_drop
#     model = NeuralClassifier(
#         net_type,
#         output_size=nC,
#         hidden_size=hidden,
#         vocab_size=vocabsize,
#         learnable_length=opt.learnable,
#         pretrained=pretrained_embeddings,
#         drop_embedding_range=drop_range,
#         drop_embedding_prop=drop_prob)
#
#     model.xavier_uniform()
#     model = model.to(device)
#     if opt.tunable:
#         model.finetune_pretrained()
#
#     return model

def init_Net(nC, vocabsize, pretrained_embeddings, sup_range, device):
    net_type=opt.net
    hidden = opt.channels if net_type == 'cnn' else opt.hidden
    if opt.droptype == 'sup':
        drop_range = sup_range
    elif opt.droptype == 'learn':
        drop_range = [pretrained_embeddings.shape[1], pretrained_embeddings.shape[1]+opt.learnable]
    elif opt.droptype == 'none':
        drop_range = None
    elif opt.droptype == 'full':
        drop_range = [0, pretrained_embeddings.shape[1]+opt.learnable]

    print('droptype =', opt.droptype)
    print('droprange =', drop_range)
    print('dropprob =', opt.dropprob)

    model = NeuralClassifier(
        net_type,
        output_size=nC,
        hidden_size=hidden,
        vocab_size=vocabsize,
        learnable_length=opt.learnable,
        pretrained=pretrained_embeddings,
        drop_embedding_range=drop_range,
        drop_embedding_prop=opt.dropprob)

    model.xavier_uniform()
    model = model.to(device)
    if opt.tunable:
        model.finetune_pretrained()

    return model


def set_method_name():
    method_name = opt.net

    if opt.pretrained:
        method_name += f'-{opt.pretrained}'
    if opt.learnable > 0:
        method_name += f'-learn{opt.learnable}'
    if opt.supervised:
        sup_drop = 0 if opt.droptype != 'sup' else opt.dropprob
        method_name += f'-supervised-d{sup_drop}-{opt.supervised_method}'
    if opt.dropprob > 0:
        if opt.droptype != 'sup':
            method_name += f'-Drop{opt.droptype}{opt.dropprob}'
    if (opt.pretrained or opt.supervised) and opt.tunable:
        method_name+='-tunable'
    if opt.weight_decay > 0:
        method_name+=f'_wd{opt.weight_decay}'
    if opt.net in {'lstm', 'attn'}:
        method_name+=f'-h{opt.hidden}'
    if opt.net== 'cnn':
        method_name+=f'-ch{opt.channels}'
    return method_name


def index_dataset(dataset, pretrained=None):
    # build the vocabulary
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


def embedding_matrix(dataset, pretrained, vocabsize, word2index, out_of_vocabulary, opt):
    print('[embedding matrix]')
    pretrained_embeddings = None
    sup_range = None
    if opt.pretrained or opt.supervised:
        pretrained_embeddings = []

        if pretrained is not None:
            word_list = get_word_list(word2index, out_of_vocabulary)
            weights = pretrained.extract(word_list)
            pretrained_embeddings.append(weights)
            print('\t[pretrained-matrix] ', weights.shape)
            del pretrained

        if opt.supervised:
            Xtr, _ = dataset.vectorize()
            Ytr = dataset.devel_labelmatrix
            F = get_supervised_embeddings(Xtr, Ytr,
                                          method=opt.supervised_method,
                                          max_label_space=opt.max_label_space,
                                          dozscore=(not opt.nozscore))
            num_missing_rows = vocabsize - F.shape[0]
            F = np.vstack((F, np.zeros(shape=(num_missing_rows, F.shape[1]))))
            F = torch.from_numpy(F).float()
            print('\t[supervised-matrix]', F.shape)

            offset = 0
            if pretrained_embeddings:
                offset = pretrained_embeddings[0].shape[1]
            sup_range = [offset, offset + F.shape[1]]
            pretrained_embeddings.append(F)

        pretrained_embeddings = torch.cat(pretrained_embeddings, dim=1)

    print('[embedding matrix done]')
    return pretrained_embeddings, sup_range


def init_optimizer(model, lr, weight_decay):
    return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)


def init_logfile(method_name, opt):
    logfile = CSVLog(opt.log_file, ['dataset', 'method', 'epoch', 'measure', 'value', 'run', 'timelapse'])
    logfile.set_default('dataset', opt.dataset)
    logfile.set_default('run', opt.seed)
    logfile.set_default('method', method_name)
    assert opt.force or not logfile.already_calculated(), f'results for dataset {opt.dataset} method {method_name} and run {opt.seed} already calculated'
    return logfile


def load_pretrained(opt):
    if opt.pretrained == 'glove':
        return GloVe(path=opt.glove_path)
    elif opt.pretrained == 'word2vec':
        return Word2Vec(path=opt.word2vec_path, limit=1000000)
    elif opt.pretrained == 'fasttext':
        return FastTextEmbeddings(path=opt.fasttext_path, limit=1000000)
    return None


def init_loss(classification_type):
    assert classification_type in ['multilabel','singlelabel'], 'unknown classification mode'
    L = torch.nn.BCEWithLogitsLoss() if classification_type == 'multilabel' else torch.nn.CrossEntropyLoss()
    return L.cuda()


def main(opt):
    method_name = set_method_name()
    logfile = init_logfile(method_name, opt)
    pretrained = load_pretrained(opt)

    dataset = Dataset.load(dataset_name=opt.dataset, pickle_path=opt.pickle_path).show()
    word2index, out_of_vocabulary, unk_index, pad_index, devel_index, test_index = index_dataset(dataset, pretrained)

    # dataset split tr/val/test
    val_size = min(int(len(devel_index) * .2), 20000)
    train_index, val_index, ytr, yval = train_test_split(
        devel_index, dataset.devel_target, test_size=val_size, random_state=opt.seed, shuffle=True
    )
    yte = dataset.test_target

    vocabsize = len(word2index) + len(out_of_vocabulary)
    pretrained_embeddings, sup_range = embedding_matrix(dataset, pretrained, vocabsize, word2index, out_of_vocabulary, opt)

    model = init_Net(dataset.nC, vocabsize, pretrained_embeddings, sup_range, opt.device)
    optim = init_optimizer(model, lr=opt.lr, weight_decay=opt.weight_decay)
    criterion = init_loss(dataset.classification_type)

    # train-validate
    tinit = time()
    create_if_not_exist(opt.checkpoint_dir)
    early_stop = EarlyStopping(model, patience=opt.patience, checkpoint=f'{opt.checkpoint_dir}/{opt.net}-{opt.dataset}')

    for epoch in range(1, opt.nepochs + 1):
        train(model, train_index, ytr, pad_index, tinit, logfile, criterion, optim, epoch, method_name)

        # validation
        macrof1 = test(model, val_index, yval, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'va')
        early_stop(macrof1, epoch)
        if opt.test_each>0:
            if (opt.plotmode and (epoch==1 or epoch%opt.test_each==0)) or (not opt.plotmode and epoch%opt.test_each==0 and epoch<opt.nepochs):
                test(model, test_index, yte, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'te')

        if early_stop.STOP:
            print('[early-stop]')
            if not opt.plotmode: # with plotmode activated, early-stop is ignored
                break

    # restores the best model according to the Mf1 of the validation set (only when plotmode==False)
    stoptime = early_stop.stop_time - tinit
    stopepoch = early_stop.best_epoch
    logfile.add_row(epoch=stopepoch, measure=f'early-stop', value=early_stop.best_score, timelapse=stoptime)

    if not opt.plotmode:
        print('performing final evaluation')
        model = early_stop.restore_checkpoint()

        if opt.val_epochs>0:
            print(f'last {opt.val_epochs} epochs on the validation set')
            for val_epoch in range(1, opt.val_epochs + 1):
                train(model, val_index, yval, pad_index, tinit, logfile, criterion, optim, epoch+val_epoch, method_name)

        # test
        print('Training complete: testing')
        test(model, test_index, yte, pad_index, dataset.classification_type, tinit, epoch, logfile, criterion, 'final-te')


def train(model, train_index, ytr, pad_index, tinit, logfile, criterion, optim, epoch, method_name):
    as_long = isinstance(criterion, torch.nn.CrossEntropyLoss)
    loss_history = []
    if opt.max_epoch_length is not None: # consider an epoch over after max_epoch_length
        tr_len = len(train_index)
        train_for = opt.max_epoch_length*opt.batch_size
        from_, to_= (epoch*train_for) % tr_len, ((epoch+1)*train_for) % tr_len
        print(f'epoch from {from_} to {to_}')
        if to_ < from_:
            train_index = train_index[from_:] + train_index[:to_]
            if issparse(ytr):
                ytr = vstack((ytr[from_:], ytr[:to_]))
            else:
                ytr = np.concatenate((ytr[from_:], ytr[:to_]))
        else:
            train_index = train_index[from_:to_]
            ytr = ytr[from_:to_]

    model.train()
    for idx, (batch, target) in enumerate(batchify(train_index, ytr, opt.batch_size, pad_index, opt.device, as_long)):
        optim.zero_grad()
        loss = criterion(model(batch), target)
        loss.backward()
        clip_gradient(model)
        optim.step()
        loss_history.append(loss.item())

        if idx % opt.log_interval == 0:
            interval_loss = np.mean(loss_history[-opt.log_interval:])
            print(f'{opt.dataset} {method_name} Epoch: {epoch}, Step: {idx}, Training Loss: {interval_loss:.6f}')

    mean_loss = np.mean(interval_loss)
    logfile.add_row(epoch=epoch, measure='tr_loss', value=mean_loss, timelapse=time() - tinit)
    return mean_loss


def test(model, test_index, yte, pad_index, classification_type, tinit, epoch, logfile, criterion, measure_prefix):
    model.eval()
    predictions = []
    target_long = isinstance(criterion, torch.nn.CrossEntropyLoss)
    for batch, target in tqdm(
            batchify(test_index, yte, opt.batch_size_test, pad_index, opt.device, target_long=target_long),
            desc='evaluation: '
    ):
        logits = model(batch)
        loss = criterion(logits, target).item()
        prediction = csr_matrix(predict(logits, classification_type=classification_type))
        predictions.append(prediction)

    yte_ = scipy.sparse.vstack(predictions)
    Mf1, mf1, acc = evaluation(yte, yte_, classification_type)
    print(f'[{measure_prefix}] Macro-F1={Mf1:.3f} Micro-F1={mf1:.3f} Accuracy={acc:.3f}')
    tend = time() - tinit

    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-macro-F1', value=Mf1, timelapse=tend)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-micro-F1', value=mf1, timelapse=tend)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-accuracy', value=acc, timelapse=tend)
    logfile.add_row(epoch=epoch, measure=f'{measure_prefix}-loss', value=loss, timelapse=tend)

    return Mf1


if __name__ == '__main__':
    available_datasets = Dataset.dataset_available
    available_dropouts = {'sup','none','full','learn'}

    # Training settings
    parser = argparse.ArgumentParser(description='Neural text classification with Word-Class Embeddings')
    parser.add_argument('--dataset', type=str, default='reuters21578', metavar='str',
                        help=f'dataset, one in {available_datasets}')
    parser.add_argument('--batch-size', type=int, default=100, metavar='int',
                        help='input batch size (default: 100)')
    parser.add_argument('--batch-size-test', type=int, default=250, metavar='int',
                        help='batch size for testing (default: 250)')
    parser.add_argument('--nepochs', type=int, default=200, metavar='int',
                        help='number of epochs (default: 200)')
    parser.add_argument('--patience', type=int, default=10, metavar='int',
                        help='patience for early-stop (default: 10)')
    parser.add_argument('--plotmode', action='store_true', default=False,
                        help='in plot mode, executes a long run in order to generate enough data to produce trend plots'
                             ' (test-each should be >0. This mode is used to produce plots, and does not perform a '
                             'final evaluation on the test set other than those performed after test-each epochs).')
    parser.add_argument('--hidden', type=int, default=512, metavar='int',
                        help='hidden lstm size (default: 512)')
    parser.add_argument('--channels', type=int, default=256, metavar='int',
                        help='number of cnn out-channels (default: 256)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='float',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0, metavar='float',
                        help='weight decay (default: 0)')
    parser.add_argument('--droptype', type=str, default='sup', metavar='DROPTYPE',
                        help=f'chooses the type of dropout to apply after the embedding layer. Default is "sup" which '
                             f'only applies to word-class embeddings (if present). Other options include "none" which '
                             f'does not apply dropout (same as "sup" with no supervised embeddings), "full" which '
                             f'applies dropout to the entire embedding, or "learn" that applies dropout only to the '
                             f'learnable embedding.')
    parser.add_argument('--dropprob', type=float, default=0.5, metavar='[0.0, 1.0]',
                        help='dropout probability (default: 0.5)')
    # parser.add_argument('--sup-drop', type=float, default=0.5, metavar='[0.0, 1.0]',
    #                     help='dropout probability for the supervised matrix (default: 0.5)')
    # parser.add_argument('--embedding-drop', type=float, default=0.0, metavar='[0.0, 1.0]',
    #                     help='dropout probability for the entire embedding matrix; if specified>0, ignores the value '
    #                          'of --sup-drop (default: 0.0, i.e., deactivated)')
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
    parser.add_argument('--net', type=str, default='lstm', metavar='str',
                        help=f'net, one in {NeuralClassifier.ALLOWED_NETS}')
    parser.add_argument('--pretrained', type=str, default=None, metavar='glove|word2vec|fasttext',
                        help='pretrained embeddings, use "glove", "word2vec", or "fasttext" (default None)')
    parser.add_argument('--supervised', action='store_true', default=False,
                        help='use supervised embeddings')
    parser.add_argument('--supervised-method', type=str, default='dotn', metavar='dotn|ppmi|ig|chi',
                        help='method used to create the supervised matrix. Available methods include dotn (default), '
                             'ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)')
    parser.add_argument('--learnable', type=int, default=0, metavar='int',
                        help='dimension of the learnable embeddings (default 0)')
    parser.add_argument('--val-epochs', type=int, default=1, metavar='int',
                        help='number of training epochs to perform on the validation set once training is '
                             'over (default 1)')
    parser.add_argument('--word2vec-path', type=str, default='../datasets/Word2Vec/GoogleNews-vectors-negative300.bin',
                        metavar='str',
                        help=f'path to GoogleNews-vectors-negative300.bin pretrained vectors (used only '
                             f'with --pretrained word2vec)')
    parser.add_argument('--glove-path', type=str, default='.vector_cache',
                        metavar='PATH',
                        help=f'path to glove.840B.300d pretrained vectors (used only with --pretrained glove)')
    parser.add_argument('--fasttext-path', type=str, default='../datasets/fastText/crawl-300d-2M.vec',
                        help=f'path to glove.840B.300d pretrained vectors (used only with --pretrained word2vec)')
    parser.add_argument('--max-label-space', type=int, default=300, metavar='int',
                        help='larger dimension allowed for the feature-label embedding (if larger, then PCA with this '
                             'number of components is applied (default 300)')
    parser.add_argument('--max-epoch-length', type=int, default=None, metavar='int',
                        help='number of (batched) training steps before considering an epoch over (None: full epoch)') #300 for wipo-sl-sc
    parser.add_argument('--force', action='store_true', default=False,
                        help='do not check if this experiment has already been run')
    parser.add_argument('--tunable', action='store_true', default=False,
                        help='pretrained embeddings are tunable from the beginning (default False, i.e., static)')
    parser.add_argument('--nozscore', action='store_true', default=False,
                        help='disables z-scoring form the computation of WCE')

    opt = parser.parse_args()

    opt.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'running on {opt.device}')
    assert f'{opt.device}'=='cuda', 'forced cuda device but cpu found'
    torch.manual_seed(opt.seed)

    assert opt.dataset in available_datasets, \
        f'unknown dataset {opt.dataset}'
    assert opt.pretrained in [None]+AVAILABLE_PRETRAINED, \
        f'unknown pretrained set {opt.pretrained}'
    assert not opt.plotmode or opt.test_each > 0, \
        'plot mode implies --test-each>0'
    assert opt.supervised_method in STWFUNCTIONS, \
        f'unknown supervised term weighting function; allowed are {STWFUNCTIONS}'
    assert opt.droptype in available_dropouts, \
        f'unknown dropout type; allowed are {available_dropouts}'
    if opt.droptype == 'sup' and opt.supervised==False:
        opt.droptype = 'none'
        print('warning: droptype="sup" but supervised="False"; the droptype changed to "none"')
    if opt.droptype == 'learn' and opt.learnable==0:
        opt.droptype = 'none'
        print('warning: droptype="learn" but learnable=0; the droptype changed to "none"')

    if opt.pickle_dir:
        opt.pickle_path = join(opt.pickle_dir, f'{opt.dataset}.pickle')

    main(opt)
