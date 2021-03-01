# Word-Class Embeddings for Multiclass Text Classification

Code to reproduce the experiments reported in the paper "Word-Class Embeddings for Multiclass Text Classification" (publised on 
[Data Mining and Knowledge Discovery, 2021](https://link.springer.com/article/10.1007/s10618-020-00735-3) -- a preprint is available [here](https://arxiv.org/abs/1911.11506)). This repo also includes a script to extract the word-class embedding matrix from any dataset so that you can use it in your model.

Word-Class Embeddings (WCEs) are a form of supervised embeddings specially suited for multiclass text classification.
WCEs are meant to be used as extensions (i.e., by concatenation) to pre-trained embeddings (e.g., GloVe or word2vec) embeddings
in order to improve the performance of neural classifiers.

## Requirements

* Pytorch (1.1.0) and torchtext (0.3.0)
* Cuda 10
* Scikit-learn (0.21.1)
* NumPy (1.16.3)
* SciPy (1.2.1)
* Pandas (0.24.2) 
* fastText (0.2.0)
* transformers
* simpletransformers

## Generating a Word-Class Embedding matrix
The script _learn_word_class_embeddings.py_ generates the WCE matrix from any dataset. The dataset must be in _fastText_ format (i.e., one already-preprocessed document for each line with labels indicated by a prefix `__label__<labelname>`). The WCE matrix is stored in disk in txt format (`<word> <dim1> <dim2> ... <dimn>\n`); support for .bin files will be added soon.
	
```
usage: learn_word_class_embeddings.py [-h] [-o OUTPUT] [-m METHOD] [-f MINFREQ]
                                     [-d MAXDIM] [-l LABEL] -i INPUT

Learn Word-Class Embeddings from dataset in fastText format

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file name where to store the WCE matrix in .txt
                        format
  -m METHOD, --method METHOD
                        Correlation method [dotn]
  -f MINFREQ, --minfreq MINFREQ
                        Minimum number of occurrences of terms [5]
  -d MAXDIM, --maxdim MAXDIM
                        Maximum number of dimensions (by default, WCEs have
                        one dimension per category) after which PCA is applied
                        [300]
  -l LABEL, --label LABEL
                        Label prefix (default __label__)

required named arguments:
  -i INPUT, --input INPUT
                        Input file path
```

## Experiments

Although all datasets used in the experiments are publicly available, not all can be download automatically.
In such cases, you will be warned with an error message explaining where and how to ask for permission. 
If otherwise, the dataset will be fetched and processed automatically.

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
               [--droptype DROPTYPE] [--dropprob [0.0, 1.0]] [--seed int]
               [--log-interval int] [--log-file str] [--pickle-dir str]
               [--test-each int] [--checkpoint-dir str] [--net str]
               [--pretrained glove|word2vec|fasttext] [--supervised]
               [--supervised-method dotn|ppmi|ig|chi] [--learnable int]
               [--val-epochs int] [--word2vec-path str] [--glove-path PATH]
               [--fasttext-path FASTTEXT_PATH] [--max-label-space int]
               [--max-epoch-length int] [--force] [--tunable] [--nozscore]

Neural text classification with Word-Class Embeddings

optional arguments:
  -h, --help            show this help message and exit
  --dataset str         dataset, one in {'wipo-sl-sc', 'jrcall', 'ohsumed',
                        'rcv1', '20newsgroups', 'wipo-sl-mg', 'reuters21578',
                        'wipo-ml-mg', 'wipo-ml-sc'}
  --batch-size int      input batch size (default: 100)
  --batch-size-test int
                        batch size for testing (default: 250)
  --nepochs int         number of epochs (default: 200)
  --patience int        patience for early-stop (default: 10)
  --plotmode            in plot mode, executes a long run in order to generate
                        enough data to produce trend plots (test-each should
                        be >0. This mode is used to produce plots, and does
                        not perform a final evaluation on the test set other
                        than those performed after test-each epochs).
  --hidden int          hidden lstm size (default: 512)
  --channels int        number of cnn out-channels (default: 256)
  --lr float            learning rate (default: 1e-3)
  --weight_decay float  weight decay (default: 0)
  --droptype DROPTYPE   chooses the type of dropout to apply after the
                        embedding layer. Default is "sup" which only applies
                        to word-class embeddings (if present). Other options
                        include "none" which does not apply dropout (same as
                        "sup" with no supervised embeddings), "full" which
                        applies dropout to the entire embedding, or "learn"
                        that applies dropout only to the learnable embedding.
  --dropprob [0.0, 1.0]
                        dropout probability (default: 0.5)
  --seed int            random seed (default: 1)
  --log-interval int    how many batches to wait before printing training
                        status
  --log-file str        path to the log csv file
  --pickle-dir str      if set, specifies the path where to save/load the
                        dataset pickled (set to None if you prefer not to
                        retain the pickle file)
  --test-each int       how many epochs to wait before invoking test (default:
                        0, only at the end)
  --checkpoint-dir str  path to the directory containing checkpoints
  --net str             net, one in {'lstm', 'cnn', 'attn'}
  --pretrained glove|word2vec|fasttext
                        pretrained embeddings, use "glove", "word2vec", or
                        "fasttext" (default None)
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
  --glove-path PATH     path to glove.840B.300d pretrained vectors (used only
                        with --pretrained glove)
  --fasttext-path FASTTEXT_PATH
                        path to glove.840B.300d pretrained vectors (used only
                        with --pretrained word2vec)
  --max-label-space int
                        larger dimension allowed for the feature-label
                        embedding (if larger, then PCA with this number of
                        components is applied (default 300)
  --max-epoch-length int
                        number of (batched) training steps before considering
                        an epoch over (None: full epoch)
  --force               do not check if this experiment has already been run
  --tunable             pretrained embeddings are tunable from the beginning
                        (default False, i.e., static)
  --nozscore            disables z-scoring form the computation of WCE
```

For example the following command:

```
python main.py --dataset 20newsgroups --net cnn --channels 256 --pretrained glove --supervised --log-file ../log/results.csv
```

Will run an experiment for the dataset _20 Newsgroups_ using CNN trained on static embeddings initialized as the concatenation
of pre-trained _GloVe_ vectors and supervised _WCEs_. The output printed on stdout looks like:

```
Loading GloVe pretrained vectors from torchtext
Done
loading pickled dataset from ../pickles/20newsgroups.pickle
[Done]
singlelabel, nD=18846=(11314+7532), nF=17184, nC=20
[unk = 67822/2198726=3.08%][out = 79947/2198726=3.64%]: 100%|██████████| 11314/11314 [00:02<00:00, 4594.88it/s]
[unk = 32785/1303459=2.52%][out = 59887/1303459=4.59%]: 100%|██████████| 7532/7532 [00:01<00:00, 6041.34it/s]
[indexing complete]
[embedding matrix]
	[pretrained-matrix]
	[supervised-matrix]
computing supervised embeddings...
[embedding matrix done]
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 1, Step: 0, Training Loss: 2.992394
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 1, Step: 10, Training Loss: 2.577988
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 1, Step: 20, Training Loss: 1.817676
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 1, Step: 30, Training Loss: 1.395195
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 1, Step: 40, Training Loss: 1.212901
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 1, Step: 50, Training Loss: 1.094817
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 1, Step: 60, Training Loss: 1.043492
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 1, Step: 70, Training Loss: 0.879524
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 1, Step: 80, Training Loss: 0.920604
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 1, Step: 90, Training Loss: 0.858984
evaluation: : 10it [00:00, 19.44it/s]
[va] Macro-F1=0.793 Micro-F1=0.797 Accuracy=0.797
[early-stop] improved, saving model in ../checkpoint/cnn-20newsgroups
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 2, Step: 0, Training Loss: 0.838249
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 2, Step: 10, Training Loss: 0.772314
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 2, Step: 20, Training Loss: 0.765503

[...]

20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 22, Step: 80, Training Loss: 0.176321
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 22, Step: 90, Training Loss: 0.169704
evaluation: : 10it [00:00, 20.27it/s]
[va] Macro-F1=0.837 Micro-F1=0.838 Accuracy=0.838
[early-stop] patience exhausted
[early-stop]
performing final evaluation
last 1 epochs on the validation set
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 23, Step: 0, Training Loss: 0.939111
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 23, Step: 10, Training Loss: 0.925299
20newsgroups cnn-glove-supervised-d0.5-dotn-ch256 Epoch: 23, Step: 20, Training Loss: 0.901256
evaluation: : 0it [00:00, ?it/s]Training complete: testing
evaluation: : 31it [00:01, 18.94it/s]
[final-te] Macro-F1=0.701 Micro-F1=0.712 Accuracy=0.712
```

This information is dumped into a csv file _../log/results.csv_ at the end of each epoch.
The csv should look like:

| dataset | epoch | measure | method | run | timelapse | value |
| ------| ------| ------| ------| ------| ------| ------|
20newsgroups | 1 | tr_loss | cnn-glove-supervised-d0.5-dotn-ch256 | 1 | 3.414764642715454 | 0.8589843213558197 |
20newsgroups | 1 | va-macro-F1 | cnn-glove-supervised-d0.5-dotn-ch256 | 1 | 3.9400274753570557 | 0.7925198931977147 |
20newsgroups | 1 | va-micro-F1 | cnn-glove-supervised-d0.5-dotn-ch256 | 1 | 3.9400274753570557 | 0.7966401414677275 |
20newsgroups | 1 | va-accuracy | cnn-glove-supervised-d0.5-dotn-ch256 | 1 | 3.9400274753570557 | 0.7966401414677277 |
20newsgroups | 1 | va-loss | cnn-glove-supervised-d0.5-dotn-ch256 | 1 | 3.9400274753570557 | 0.5694978833198547 |
20newsgroups | 2 | tr_loss | cnn-glove-supervised-d0.5-dotn-ch256 | 1 | 7.470069885253906 | 0.6080582588911057 |
20newsgroups | 2 | va-macro-F1 | cnn-glove-supervised-d0.5-dotn-ch256 | 1 | 8.01769471168518 | 0.8235132472427036 |
20newsgroups | 2 | va-micro-F1 | cnn-glove-supervised-d0.5-dotn-ch256 | 1 | 8.01769471168518 | 0.8249336870026526 |
20newsgroups | 2 | va-accuracy | cnn-glove-supervised-d0.5-dotn-ch256 | 1 | 8.01769471168518 | 0.8249336870026526 |
20newsgroups | 2 | va-loss | cnn-glove-supervised-d0.5-dotn-ch256 | 1 | 8.01769471168518 | 0.42492079734802246 |
... | ... | ... | ... | ... | ... | ... | ... |
20newsgroups | 12 | early-stop | cnn-glove-supervised-d0.5-dotn-ch256 | 1 | 45.899415254592896 | 0.8421921175935061 |
20newsgroups | 22 | tr_loss | cnn-glove-supervised-d0.5-dotn-ch256 | 1 | 82.66418981552124 | 0.901255750656128 |
20newsgroups | 22 | final-te-macro-F1 | cnn-glove-supervised-d0.5-dotn-ch256 | 1 | 84.31677627563477 | 0.7014112131823149 |
20newsgroups | 22 | final-te-micro-F1 | cnn-glove-supervised-d0.5-dotn-ch256 | 1 | 84.31677627563477 | 0.7124269782262347 |
20newsgroups | 22 | final-te-accuracy | cnn-glove-supervised-d0.5-dotn-ch256 | 1 | 84.31677627563477 | 0.7124269782262347 |
20newsgroups | 22 | final-te-loss | cnn-glove-supervised-d0.5-dotn-ch256 | 1 | 84.31677627563477 | 1.9046727418899536 |

Displaying the training and validation losses, along with some evaluation metrics (macro-F1, micro-F1 and accuracy) in the validation set, for each epoch. This goes on until training eventually ends (in this case, because early-stop has encountered 10 training epochs without any improvement in the validation set). The last lines of the csv account for the the final evaluation in test (after restoring the best model parameters found in epoch 12).

The script "script_10runs.sh" runs all experiments, invoking each neural architecture
with the hyperparameters found optimal during grid-search optimization. Each 
variant is run 10 times.

