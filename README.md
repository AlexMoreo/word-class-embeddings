# Word-Class Embeddings for Multiclass Text Classification

Code to reproduce the experiments reported in the paper "Word-Class Embeddings for Multiclass Text Classification" (to appear).

Word-Class Embeddings (WCEs) are a form of supervised embeddings specially suited for multiclass text classification.
WCEs are meant to be used as extensions (i.e., by concatenation) to pre-trained embeddings (e.g., GloVe or word2vec) embeddings
in order to improve the performance of neural classifiers.

##Requirements

* Pytorch (1.1.0) and torchtext (0.3.0)
* Cuda 10
* Scikit-learn (0.21.1)
* NumPy (1.16.3)
* SciPy (1.2.1)
* Pandas (0.24.2) 
* fastText (0.2.0)

Although all datasets used in the experiments are publicly available, not all can be download automatically.
In such cases, you will be warned with an error message explaining where and how to ask for permission. 
If otherwise, the dataset will be fetched and processed automatically.

## Experiments

Our experiments can be reproduced using:
* main.py: implements basic versions of three neural architectures (CNNs, LSTMs, and ATTN)
* fasttext.py: is a wrapper of the official fasttext implementation that permits to invoke it with WCEs
* svm_baselines.py: train and test of SVMs with different sets of features

Each file comes with a command line help that can be consulted by typing, e.g, 
_python main.py -h_

```
usage: main.py [-h] [--dataset str] [--batch-size int] [--batch-size-test int]
               [--nepochs int] [--patience int] [--plotmode] [--hidden int]
               [--channels int] [--lr float] [--weight_decay float]
               [--sup-drop [0.0, 1.0]] [--seed int] [--log-interval int]
               [--log-file str] [--test-each int] [--checkpoint-dir str]
               [--net str] [--pretrained glove|word2vec] [--supervised]
               [--supervised-method dotn|ppmi|ig|chi] [--learnable int]
               [--val-epochs int] [--word2vec-path str] [--glove-path str]
               [--max-label-space int] [--max-epoch-length int] [--force]
               [--tunable]

Neural text classification with Word-Class Embeddings

optional arguments:
  -h, --help            show this help message and exit
  --dataset str         dataset, one in {'wipo-ml-mg', 'reuters21578',
                        'ohsumed', 'jrcall', 'rcv1', '20newsgroups', 'wipo-sl-
                        sc', 'wipo-ml-sc', 'wipo-sl-mg'}
  --batch-size int      input batch size (default: 100)
  --batch-size-test int
                        batch size for testing (default: 250)
  --nepochs int         number of epochs (default: 200)
  --patience int        patience for early-stop (default: 10)
  --plotmode            in plot mode executes a long run in order to generate
                        enough data to produce trend plots (test-each should
                        be >0. This mode is used to produce plots, and does
                        not perform an evaluation on the test set.
  --hidden int          hidden lstm size (default: 512)
  --channels int        number of cnn out-channels (default: 128)
  --lr float            learning rate (default: 1e-3)
  --weight_decay float  weight decay (default: 0)
  --sup-drop [0.0, 1.0]
                        dropout probability for the supervised matrix
                        (default: 0.5)
  --seed int            random seed (default: 1)
  --log-interval int    how many batches to wait before printing training
                        status
  --log-file str        path to the log csv file
  --test-each int       how many epochs to wait before invoking test (default:
                        0, only at the end)
  --checkpoint-dir str  path to the directory containing checkpoints
  --net str             net, one in {'cnn', 'attn', 'lstm'}
  --pretrained glove|word2vec
                        pretrained embeddings, use "glove" or "word2vec"
                        (default None)
  --supervised          use supervised embeddings
  --supervised-method dotn|ppmi|ig|chi
                        method used to create the supervised matrix. Available
                        methods include dotn (default), ppmi (positive
                        pointwise mutual information), ig (information gain)
                        and chi (Chi-squared)
  --learnable int       dimension of the learnable embeddings (default 0)
  --val-epochs int      number of training epochs to perform on the validation
                        set once training is over (default 1)
  --word2vec-path str   path to GoogleNews-vectors-negative300.bin pretrained
                        vectors (used only with --pretrained word2vec)
  --glove-path str      path to glove.840B.300d pretrained vectors (used only
                        with --pretrained word2vec)
  --max-label-space int
                        larger dimension allowed for the feature-label
                        embedding (if larger, then PCA with this number of
                        components is applied (default 300)
  --max-epoch-length int
                        number of (batched) training steps before considering
                        an epoch over (None: full epoch)
  --force               do not check if this experiment has already been run
  --tunable             pretrained embeddings are tunable from the begining
                        (default False, i.e., static)
```



The script "script_10runs.sh" runs all experiments, invoking each neural architecture
with the hyperparameters found optimal during grid-search optimization. Each 
variant is run 10 times.

