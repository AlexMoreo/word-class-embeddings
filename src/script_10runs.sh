#!/usr/bin/env bash
source activate torch

PY="python main.py"
LOG="--log-file ../log/10runs.csv"
CNN="--net cnn"
LSTM="--net lstm"
ATTN="--net attn"

for run in {1..10} # 0 is for plots, 1 is already performed in hyper parameter search
do
dataset="--dataset 20newsgroups"
$PY $LOG $dataset	$CNN	--learnable 200	--channels 256 --seed $run
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove --seed $run
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--tunable --seed $run
$PY $LOG $dataset	$CNN	--learnable 20	--channels 128	--pretrained glove	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised --seed $run
$PY $LOG $dataset	$CNN	--channels 128	--pretrained glove	--supervised	--tunable --seed $run

$PY $LOG $dataset	$LSTM	--learnable 200	--hidden 256 --seed $run
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove --seed $run
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--tunable --seed $run
$PY $LOG $dataset	$LSTM	--learnable 20	--hidden 256	--pretrained glove	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove	--supervised --seed $run
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--supervised	--tunable --seed $run

$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256 --seed $run
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove --seed $run
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--tunable --seed $run
$PY $LOG $dataset	$ATTN	--learnable 20	--hidden 256	--pretrained glove	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised --seed $run
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run

dataset="--dataset jrcall"
$PY $LOG $dataset	$CNN	--learnable 200	--channels 512 --seed $run
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove --seed $run
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--tunable --seed $run
$PY $LOG $dataset	$CNN	--learnable 300	--channels 512	--pretrained glove	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised	--tunable --seed $run
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised --seed $run

$PY $LOG $dataset	$LSTM	--learnable 50	--hidden 1024 --seed $run
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove --seed $run
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--tunable --seed $run
$PY $LOG $dataset	$LSTM	--learnable 300	--hidden 256	--pretrained glove	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised --seed $run
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run

$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 1024	 --seed $run
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove --seed $run
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--tunable --seed $run
$PY $LOG $dataset	$ATTN	--learnable 300	--hidden 256	--pretrained glove	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained glove	--supervised --seed $run
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run

dataset="--dataset ohsumed"
$PY $LOG $dataset	$CNN	--learnable 200	--channels 512 --seed $run
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove --seed $run
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--tunable --seed $run
$PY $LOG $dataset	$CNN	--learnable 23	--channels 512	--pretrained glove	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--supervised --seed $run
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--supervised	--tunable --seed $run

$PY $LOG $dataset	$LSTM	--learnable 200 --seed $run
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove --seed $run
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--tunable --seed $run
$PY $LOG $dataset	$LSTM	--learnable 23	--hidden 1024	--pretrained glove	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--supervised	--tunable --seed $run
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove	--supervised --seed $run

$PY $LOG $dataset	$ATTN	--learnable 200	 --seed $run
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove --seed $run
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove	--tunable --seed $run
$PY $LOG $dataset	$ATTN	--learnable 23	--hidden 1024	--pretrained glove	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained glove	--supervised --seed $run
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run

dataset="--dataset rcv1"
$PY $LOG $dataset	$CNN	--learnable 200	--channels 512 --seed $run
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove --seed $run
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--tunable --seed $run
$PY $LOG $dataset	$CNN	--learnable 101	--channels 512	--pretrained glove	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--supervised --seed $run
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--supervised	--tunable --seed $run

$PY $LOG $dataset	$LSTM	--learnable 200 --seed $run
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove --seed $run
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--tunable --seed $run
$PY $LOG $dataset	$LSTM	--learnable 101	--hidden 1024	--pretrained glove	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--supervised	--tunable --seed $run
$PY $LOG $dataset	$LSTM	--hidden 2048	--pretrained glove	--supervised --seed $run

$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256 --seed $run
$PY $LOG $dataset	$ATTN	--hidden 2048	--pretrained glove --seed $run
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove	--tunable --seed $run
$PY $LOG $dataset	$ATTN	--learnable 101	--hidden 1024	--pretrained glove	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--pretrained glove	--supervised --seed $run
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained glove	--supervised	--tunable --seed $run

dataset="--dataset wipo-sl-sc"
$PY $LOG $dataset	$CNN	--learnable 200	--channels 128 --seed $run --max-epoch-length 300
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove --seed $run --max-epoch-length 300
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--tunable --seed $run --max-epoch-length 300
$PY $LOG $dataset	$CNN	--learnable 300	--channels 512	--pretrained glove	--tunable --seed $run --max-epoch-length 300 --droptype learn
$PY $LOG $dataset	$CNN	--sup-drop 0.2	--channels 512	--pretrained glove	--supervised --seed $run --max-epoch-length 300
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised	--tunable --seed $run --max-epoch-length 300

$PY $LOG $dataset	$LSTM	--learnable 200 --seed $run --max-epoch-length 300
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove --seed $run --max-epoch-length 300
$PY $LOG $dataset	$LSTM	--hidden 1024	--pretrained glove	--tunable --seed $run --max-epoch-length 300
$PY $LOG $dataset	$LSTM	--learnable 300	--hidden 1024	--pretrained glove	--tunable --seed $run --max-epoch-length 300 --droptype learn
$PY $LOG $dataset	$LSTM	--sup-drop 0.2	--pretrained glove	--supervised --seed $run --max-epoch-length 300
$PY $LOG $dataset	$LSTM	--hidden 512	--pretrained glove	--supervised	--tunable --seed $run --max-epoch-length 300

$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256	 --seed $run --max-epoch-length 300
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove --seed $run --max-epoch-length 300
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove	--tunable --seed $run --max-epoch-length 300
$PY $LOG $dataset	$ATTN	--learnable 300	--hidden 1024	--pretrained glove	--tunable --seed $run --max-epoch-length 300 --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--pretrained glove	--supervised --seed $run --max-epoch-length 300
$PY $LOG $dataset	$ATTN	--hidden 512	--pretrained glove	--supervised	--tunable --seed $run --max-epoch-length 300
#
dataset="--dataset reuters21578"
$PY $LOG $dataset	$CNN	--learnable 200	--channels 256 --seed $run
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove --seed $run
$PY $LOG $dataset	$CNN	--channels 256	--pretrained glove	--tunable --seed $run
$PY $LOG $dataset	$CNN	--learnable 115	--channels 256	--pretrained glove	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised --seed $run
$PY $LOG $dataset	$CNN	--channels 512	--pretrained glove	--supervised	--tunable --seed $run

$PY $LOG $dataset	$LSTM	--learnable 200 --seed $run
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove --seed $run
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--tunable --seed $run
$PY $LOG $dataset	$LSTM	--learnable 115	--hidden 256	--pretrained glove	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised --seed $run
$PY $LOG $dataset	$LSTM	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run

$PY $LOG $dataset	$ATTN	--learnable 200	--hidden 256 --seed $run
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove --seed $run
$PY $LOG $dataset	$ATTN	--hidden 1024	--pretrained glove	--tunable --seed $run
$PY $LOG $dataset	$ATTN	--learnable 115	--hidden 1024	--pretrained glove	--tunable --seed $run --droptype learn
$PY $LOG $dataset	$ATTN	--sup-drop 0.2	--hidden 256	--pretrained glove	--supervised --seed $run
$PY $LOG $dataset	$ATTN	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run


common="--dataset-dir ../fasttext/dataset --log-file ../log/fasttext.10runs.csv"
python fasttext.py --dataset 20newsgroups	--learnable 50	--lr 0.5	--nepochs 200	--seed $run --pickle-path ../pickles/20newsgroups.pickle
python fasttext.py --dataset jrcall	--learnable 200	--lr 0.25	--nepochs 200	--seed $run --pickle-path ../pickles/jrcall.pickle
python fasttext.py --dataset ohsumed	--learnable 200	--lr 0.5	--nepochs 200	--seed $run --pickle-path ../pickles/ohsumed.pickle
python fasttext.py --dataset rcv1	--learnable 50	--lr 0.5	--nepochs 100	--seed $run --pickle-path ../pickles/rcv1.pickle
python fasttext.py --dataset reuters21578	--learnable 200	--lr 0.5	--nepochs 100	--seed $run --pickle-path ../pickles/reuters21578.pickle
python fasttext.py --dataset wipo-sl-sc	--learnable 200	--lr 0.5	--nepochs 10	--seed $run --pickle-path ../pickles/wipo-sl-sc.pickle

done
