import argparse
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import scipy
from embedding.supervised import get_supervised_embeddings
from main import init_logfile, init_optimizer, init_loss
from model.classification import NeuralClassifier, Token2BertEmbeddings, Token2WCEmbeddings, BertWCEClassifier
from util.early_stop import EarlyStopping
from util.common import clip_gradient, predict, tokenize_parallel
from data.dataset import *
from util.csv_log import CSVLog
from util.file import create_if_not_exist
from util.metrics import *
from time import time
from embedding.pretrained import *
from scipy.sparse import vstack
import math


def init_Net(nC, bert, wce, device):
    net_type = opt.net
    hidden = opt.channels if net_type == 'cnn' else opt.hidden
    model = BertWCEClassifier(
        net_type,
        output_size=nC,
        hidden_size=hidden,
        token2bert_embeddings=bert,
        token2wce_embeddings=wce)

    model.xavier_uniform()
    model = model.to(device)
    if opt.tunable:
        model.finetune_pretrained()

    return model


def set_method_name():
    method_name = opt.net + '-bert'
    if opt.supervised:
        method_name += f'-supervised-d{opt.sup_drop}-{opt.supervised_method}'
        if opt.tunable:
            method_name+='-tunable'
    if opt.weight_decay > 0:
        method_name+=f'_wd{opt.weight_decay}'
    if opt.net in {'lstm', 'attn'}:
        method_name+=f'-h{opt.hidden}'
    if opt.net== 'cnn':
        method_name+=f'-ch{opt.channels}'
    return method_name


def embedding_matrix(opt, dataset):
    print('\t[supervised-matrix]')
    tfidf = TfidfVectorizer(tokenizer=str.split, min_df=5)
    Xtr = tfidf.fit_transform([' '.join(tokens) for tokens in dataset.devel_raw])  # already tokenized and trunctated

    print('\t[tokenization complete]')
    Ytr = dataset.devel_labelmatrix
    WCE = get_supervised_embeddings(Xtr, Ytr, method=opt.supervised_method)

    WCE_range = [0, WCE.shape[1]]
    WCE_vocab = tfidf.vocabulary_
    WCE_vocab['[UNK]'] = len(WCE_vocab)
    WCE_vocab['[PAD]'] = len(WCE_vocab)
    WCE_vocab['[SEP]'] = len(WCE_vocab)
    num_missing_rows = len(WCE_vocab) - WCE.shape[0]

    WCE = np.vstack((WCE, np.zeros(shape=(num_missing_rows, WCE.shape[1]))))
    WCE = torch.from_numpy(WCE).float()

    print(f'[supervised-matrix] shape={WCE.shape}')
    return WCE, WCE_range, WCE_vocab


def tokenize_and_truncate(dataset, tokenizer, max_length):
    dataset.devel_raw = tokenize_parallel(dataset.devel_raw, tokenizer.tokenize, max_length)
    dataset.test_raw = tokenize_parallel(dataset.test_raw, tokenizer.tokenize, max_length)


def train_val_test(dataset, seed):
    val_size = min(int(len(dataset.devel_raw) * .2), 20000)
    train_docs, val_docs, ytr, yval = train_test_split(
        dataset.devel_raw, dataset.devel_target, test_size=val_size, random_state=seed, shuffle=True
    )
    return (train_docs, ytr), (val_docs, yval), (dataset.test_raw, dataset.test_target)


def target_type(y, criterion, device):
    totype = torch.LongTensor if isinstance(criterion, torch.nn.CrossEntropyLoss) else torch.FloatTensor
    return totype(y).to(device)


def main(opt):
    method_name = set_method_name()
    logfile = init_logfile(method_name, opt)

    dataset = Dataset.load(dataset_name=opt.dataset, pickle_path=opt.pickle_path).show()
    #dataset.devel_raw=dataset.devel_raw[:100]
    #dataset.devel_target = dataset.devel_target[:100]
    #dataset.devel_labelmatrix = dataset.devel_labelmatrix[:100]

    # tokenize and truncate to max_length
    bert = Token2BertEmbeddings('bert-base-uncased', max_length=opt.max_length, device=opt.device)
    tokenize_and_truncate(dataset, bert.tokenizer, opt.max_length)

    # dataset split tr/val/test
    (train_docs, ytr), (val_docs, yval), (test_docs, yte) = train_val_test(dataset, opt.seed)

    wce = None
    if opt.supervised:
        WCE, WCE_range, WCE_vocab = embedding_matrix(opt, dataset)
        wce = Token2WCEmbeddings(
            WCE, WCE_range, WCE_vocab, drop_embedding_prop=opt.sup_drop, device=opt.device, max_length=opt.max_length
        )

    model = init_Net(dataset.nC, bert, wce, opt.device)
    optim = init_optimizer(model, lr=opt.lr, weight_decay=opt.weight_decay)
    criterion = init_loss(dataset.classification_type)

    # train-validate
    tinit = time()
    create_if_not_exist(opt.checkpoint_dir)
    early_stop = EarlyStopping(model, patience=opt.patience,
                               checkpoint=f'{opt.checkpoint_dir}/{opt.net}-{opt.dataset}' if not opt.plotmode else None)

    train_batcher = Batcher(opt.batch_size, opt.max_epoch_length)
    for epoch in range(1, opt.nepochs + 1):
        train(model, train_docs, ytr, tinit, logfile, criterion, optim, epoch, method_name, train_batcher, opt)

        # validation
        macrof1 = test(model, val_docs, yval, dataset.classification_type, tinit, epoch, logfile, criterion, 'va', opt)
        early_stop(macrof1, epoch)
        if opt.test_each>0:
            if (opt.plotmode and (epoch==1 or epoch%opt.test_each==0)) or \
                    (not opt.plotmode and epoch%opt.test_each==0 and epoch<opt.nepochs):
                test(model, test_docs, yte, dataset.classification_type, tinit, epoch, logfile, criterion, 'te', opt)

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
            val_batcher = Batcher(opt.batch_size, opt.max_epoch_length)
            for val_epoch in range(1, opt.val_epochs + 1):
                train(model, val_docs, yval, tinit, logfile, criterion, optim, epoch+val_epoch, method_name, val_batcher, opt)

        # test
        print('Training complete: testing')
        test(model, test_docs, yte, dataset.classification_type, tinit, epoch, logfile, criterion, 'final-te', opt)


class Batcher:
    def __init__(self, batchsize, max_epoch_length=None):
        self.epoch = 0
        self.batchsize = batchsize
        self.max_epoch_length = max_epoch_length
        self.offset = 0

    def batch(self, X, y):
        assert len(X) == y.shape[0], 'inconsistent sizes in args'
        concat_y = vstack if issparse(y) else np.concatenate
        n_batches = math.ceil(len(X) / self.batchsize)
        allow_circular = self.max_epoch_length is not None
        if allow_circular and (n_batches > self.max_epoch_length):
            n_batches = self.max_epoch_length
        for b in range(n_batches):
            from_ = self.offset
            to_ = (self.offset + self.batchsize) % len(X)
            if from_ < to_:
                batch_x = X[from_: to_]
                batch_y = y[from_: to_]
            else:
                if allow_circular:
                    batch_x = X[from_:] + X[: to_]
                    batch_y = concat_y((y[from_:], y[: to_]))
                else:
                    batch_x = X[from_:]
                    batch_y = y[from_:]
            if issparse(batch_y):
                batch_y = batch_y.toarray()
            yield batch_x, batch_y
            self.offset = to_


def train(model, train_index, ytr, tinit, logfile, criterion, optim, epoch, method_name, train_batch, opt):

    loss_history = []
    model.train()
    for idx, (batch, target) in enumerate(train_batch.batch(train_index, ytr)):
        optim.zero_grad()
        loss = criterion(model(batch), target_type(target, criterion, opt.device))
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


def test(model, test_docs, yte, classification_type, tinit, epoch, logfile, criterion, measure_prefix, opt):
    model.eval()
    predictions = []
    batcher = Batcher(opt.batch_size)
    with torch.no_grad():
        for batch, target in tqdm(batcher.batch(test_docs, yte), desc='evaluation: '):
            logits = model(batch)
            loss = criterion(logits, target_type(target, criterion, opt.device)).item()
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

    # Training settings
    parser = argparse.ArgumentParser(description='Neural text classification with Word-Class Embeddings')
    parser.add_argument('--dataset', type=str, default='reuters21578', metavar='str',
                        help=f'dataset, one in {available_datasets}')
    parser.add_argument('--batch-size', type=int, default=50, metavar='int',
                        help='input batch size (default: 50)')
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
    parser.add_argument('--sup-drop', type=float, default=0.5, metavar='[0.0, 1.0]',
                        help='dropout probability for the supervised matrix (default: 0.5)')
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
    parser.add_argument('--supervised', action='store_true', default=False,
                        help='use supervised embeddings')
    parser.add_argument('--supervised-method', type=str, default='dotn', metavar='dotn|ppmi|ig|chi',
                        help='method used to create the supervised matrix. Available methods include dotn (default), '
                             'ppmi (positive pointwise mutual information), ig (information gain) and chi (Chi-squared)')
    parser.add_argument('--val-epochs', type=int, default=1, metavar='int',
                        help='number of training epochs to perform on the validation set once training is '
                             'over (default 1)')
    parser.add_argument('--max-label-space', type=int, default=300, metavar='int',
                        help='larger dimension allowed for the feature-label embedding (if larger, then PCA with this '
                             'number of components is applied (default 300)')
    parser.add_argument('--max-epoch-length', type=int, default=None, metavar='int',
                        help='number of (batched) training steps before considering an epoch over (None: full epoch)') #300 for wipo-sl-sc
    parser.add_argument('--max-length', type=int, default=250, metavar='int',
                        help='max document length (in #tokens) after which a document will be cut')
    parser.add_argument('--force', action='store_true', default=False,
                        help='do not check if this experiment has already been run')
    parser.add_argument('--tunable', action='store_true', default=False,
                        help='pretrained embeddings are tunable from the beginning (default False, i.e., static)')

    opt = parser.parse_args()

    opt.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'running on {opt.device}')
    torch.manual_seed(opt.seed)

    assert opt.dataset in available_datasets, f'unknown dataset {opt.dataset}'
    assert not opt.plotmode or opt.test_each > 0, 'plot mode implies --test-each>0'
    assert opt.supervised_method in ['dotn', 'ppmi', 'ig', 'chi2']
    if opt.pickle_dir:
        opt.pickle_path = join(opt.pickle_dir, f'{opt.dataset}.pickle')

    main(opt)
