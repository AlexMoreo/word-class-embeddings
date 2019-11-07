#!/usr/bin/env bash

source activate torch

log="--log-file ../log/10runs.csv"
for run in {1..10}
do
dataset=20newsgroups
python main.py $log  --dataset $dataset	--net cnn	--learnable 200	--channels 256 --seed $run
python main.py $log  --dataset $dataset	--net cnn	--channels 128	--pretrained glove --seed $run
python main.py $log  --dataset $dataset	--net cnn	--channels 128	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net cnn	--learnable 20	--channels 128	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net cnn	--channels 128	--pretrained glove	--supervised --seed $run
python main.py $log  --dataset $dataset	--net cnn	--channels 128	--pretrained glove	--supervised	--tunable --seed $run

python main.py $log  --dataset $dataset	--net lstm	--learnable 200	--hidden 256 --seed $run
python main.py $log  --dataset $dataset	--net lstm	--hidden 256	--pretrained glove --seed $run
python main.py $log  --dataset $dataset	--net lstm	--hidden 256	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net lstm	--learnable 20	--hidden 256	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net lstm	--hidden 2048	--pretrained glove	--supervised --seed $run
python main.py $log  --dataset $dataset	--net lstm	--hidden 1024	--pretrained glove	--supervised	--tunable --seed $run

python main.py $log  --dataset $dataset	--net attn	--learnable 200	--hidden 256 --seed $run
python main.py $log  --dataset $dataset	--net attn	--hidden 256	--pretrained glove --seed $run
python main.py $log  --dataset $dataset	--net attn	--hidden 256	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net attn	--learnable 20	--hidden 256	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net attn	--hidden 256	--pretrained glove	--supervised --seed $run
python main.py $log  --dataset $dataset	--net attn	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run

dataset=jrcall
python main.py $log  --dataset $dataset	--net cnn	--learnable 200	--channels 512 --seed $run
python main.py $log  --dataset $dataset	--net cnn	--channels 512	--pretrained glove --seed $run
python main.py $log  --dataset $dataset	--net cnn	--channels 512	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net cnn	--learnable 300	--channels 512	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net cnn	--channels 256	--pretrained glove	--supervised    --seed $run
python main.py $log  --dataset $dataset	--net cnn	--channels 256	--pretrained glove	--supervised    --tunable --seed $run

python main.py $log  --dataset $dataset	--net lstm	--learnable 50	--hidden 1024 --seed $run
python main.py $log  --dataset $dataset	--net lstm	--hidden 256	--pretrained glove --seed $run
python main.py $log  --dataset $dataset	--net lstm	--hidden 256	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net lstm	--learnable 300	--hidden 256	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net lstm	--hidden 256	--pretrained glove	--supervised    --seed $run
python main.py $log  --dataset $dataset	--net lstm	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run

python main.py $log  --dataset $dataset	--net attn	--learnable 200	--hidden 1024	 --seed $run
python main.py $log  --dataset $dataset	--net attn	--hidden 256	--pretrained glove --seed $run
python main.py $log  --dataset $dataset	--net attn	--hidden 256	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net attn	--learnable 300	--hidden 256	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net attn	--hidden 256	--pretrained glove	--supervised    --seed $run
python main.py $log  --dataset $dataset	--net attn	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run

dataset=ohsumed
python main.py $log  --dataset $dataset	--net cnn	--learnable 200	--channels 512 --seed $run
python main.py $log  --dataset $dataset	--net cnn	--channels 512	--pretrained glove --seed $run
python main.py $log  --dataset $dataset	--net cnn	--channels 512	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net cnn	--learnable 23	--channels 512	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net cnn	--channels 256	--pretrained glove	--supervised --seed $run
python main.py $log  --dataset $dataset	--net cnn	--channels 256	--pretrained glove	--supervised	--tunable --seed $run

python main.py $log  --dataset $dataset	--net lstm	--learnable 200 --seed $run
python main.py $log  --dataset $dataset	--net lstm	--hidden 2048	--pretrained glove --seed $run
python main.py $log  --dataset $dataset	--net lstm	--hidden 1024	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net lstm	--learnable 23	--hidden 1024	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net lstm	--hidden 1024	--pretrained glove	--supervised	--tunable --seed $run
python main.py $log  --dataset $dataset	--net lstm	--hidden 2048	--pretrained glove	--supervised --seed $run

python main.py $log  --dataset $dataset	--net attn	--learnable 200	 --seed $run
python main.py $log  --dataset $dataset	--net attn	--hidden 1024	--pretrained glove --seed $run
python main.py $log  --dataset $dataset	--net attn	--learnable 23	--hidden 1024	--pretrained glove	--tunable --seed $run
python main.py $log  --dataset $dataset	--net attn	--sup-drop 0.2	--hidden 256	--pretrained glove	--supervised --seed $run
python main.py $log  --dataset $dataset	--net attn	--hidden 256	--pretrained glove	--supervised	--tunable --seed $run

dataset=rcv1
python main.py $log  --dataset $dataset	--net cnn	--learnable 200	--channels 512 --seed $run --fullpickle ../pickles/$dataset-noglove-index.pickle
python main.py $log  --dataset $dataset	--net cnn	--channels 512	--pretrained glove --seed $run --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net cnn	--channels 512	--pretrained glove	--tunable --seed $run  --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net cnn	--learnable 101	--channels 512	--pretrained glove	--tunable --seed $run --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net cnn	--channels 256	--pretrained glove	--supervised --seed $run --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net cnn	--channels 256	--pretrained glove	--supervised	--tunable --seed $run --fullpickle ../pickles/$dataset-index.pickle

python main.py $log  --dataset $dataset	--net lstm	--learnable 200 --seed $run --fullpickle ../pickles/$dataset-noglove-index.pickle
python main.py $log  --dataset $dataset	--net lstm	--hidden 2048	--pretrained glove --seed $run --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net lstm	--hidden 1024	--pretrained glove	--tunable --seed $run --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net lstm	--learnable 101	--hidden 1024	--pretrained glove	--tunable --seed $run --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net lstm	--hidden 1024	--pretrained glove	--supervised	--tunable --seed $run --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net lstm	--hidden 2048	--pretrained glove	--supervised --seed $run --fullpickle ../pickles/$dataset-index.pickle

python main.py $log  --dataset $dataset	--net attn	--learnable 200	--hidden 256 --seed $run --fullpickle ../pickles/$dataset-noglove-index.pickle
python main.py $log  --dataset $dataset	--net attn	--hidden 2048	--pretrained glove --seed $run --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net attn	--hidden 1024	--pretrained glove	--tunable --seed $run --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net attn	--learnable 101	--hidden 1024	--pretrained glove	--tunable --seed $run --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net attn	--sup-drop 0.2	--pretrained glove	--supervised --seed $run --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net attn	--hidden 512	--pretrained glove	--supervised	--tunable --seed $run --fullpickle ../pickles/$dataset-index.pickle

dataset=wipo-sl-sc
python main.py $log  --dataset $dataset	--net cnn	--learnable 200	--channels 128 --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-noglove-index.pickle
python main.py $log  --dataset $dataset	--net cnn	--channels 512	--pretrained glove --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net cnn	--channels 512	--pretrained glove	--tunable --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net cnn	--learnable 300	--channels 512	--pretrained glove	--tunable --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net cnn	--sup-drop 0.2	--channels 128	--pretrained glove	--supervised --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net cnn	--sup-drop 0.2  --channels 128	--pretrained glove	--supervised	--tunable --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-index.pickle

python main.py $log  --dataset $dataset	--net lstm	--learnable 200 --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-noglove-index.pickle
python main.py $log  --dataset $dataset	--net lstm	--hidden 1024	--pretrained glove --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net lstm	--hidden 1024	--pretrained glove	--tunable --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net lstm	--learnable 300	--hidden 1024	--pretrained glove	--tunable --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net lstm	--sup-drop 0.2	--hidden 512    --pretrained glove	--supervised --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net lstm	--sup-drop 0.2  --hidden 512	--pretrained glove	--supervised	--tunable --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-index.pickle

python main.py $log  --dataset $dataset	--net attn	--learnable 200	--hidden 256	 --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-noglove-index.pickle
python main.py $log  --dataset $dataset	--net attn	--hidden 1024	--pretrained glove --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net attn	--hidden 1024	--pretrained glove	--tunable --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net attn	--learnable 300	--hidden 1024	--pretrained glove	--tunable --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net attn	--sup-drop 0.2	--hidden 512    --pretrained glove	--supervised --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-index.pickle
python main.py $log  --dataset $dataset	--net attn	--sup-drop 0.2  --hidden 512	--pretrained glove	--supervised	--tunable --seed $run --max-epoch-length 300 --fullpickle ../pickles/$dataset-index.pickle
#
dataset=reuters21578
python main.py $log  --dataset $dataset	--net cnn	--learnable 200	--channels 256	 --seed $run
python main.py $log  --dataset $dataset	--net cnn	--channels 256	--pretrained glove	 --seed $run
python main.py $log  --dataset $dataset	--net cnn	--channels 256	--pretrained glove	--tunable	 --seed $run
python main.py $log  --dataset $dataset	--net cnn	--learnable 115	--channels 256	--pretrained glove	--tunable	 --seed $run
python main.py $log  --dataset $dataset	--net cnn	--channels 512	--pretrained glove	--supervised	 --seed $run
python main.py $log  --dataset $dataset	--net cnn	--channels 512	--pretrained glove	--supervised	--tunable	 --seed $run

python main.py $log  --dataset $dataset	--net lstm	--learnable 200	 --seed $run
python main.py $log  --dataset $dataset	--net lstm	--hidden 256	--pretrained glove	 --seed $run
python main.py $log  --dataset $dataset	--net lstm	--hidden 256	--pretrained glove	--tunable	 --seed $run
python main.py $log  --dataset $dataset	--net lstm	--learnable 115	--hidden 256	--pretrained glove	--tunable	 --seed $run
python main.py $log  --dataset $dataset	--net lstm	--hidden 256	--pretrained glove	--supervised	 --seed $run
python main.py $log  --dataset $dataset	--net lstm	--hidden 256	--pretrained glove	--supervised	--tunable	 --seed $run

python main.py $log  --dataset $dataset	--net attn	--learnable 200	--hidden 256	 --seed $run
python main.py $log  --dataset $dataset	--net attn	--hidden 1024	--pretrained glove	 --seed $run
python main.py $log  --dataset $dataset	--net attn	--hidden 1024	--pretrained glove	--tunable	 --seed $run
python main.py $log  --dataset $dataset	--net attn	--learnable 115	--hidden 1024	--pretrained glove	--tunable	 --seed $run
python main.py $log  --dataset $dataset	--net attn	--sup-drop 0.2	--hidden 256	--pretrained glove	--supervised	 --seed $run
python main.py $log  --dataset $dataset	--net attn	--hidden 256	--pretrained glove	--supervised	--tunable	 --seed $run


common="--dataset-dir ../fasttext/dataset --log-file ../log/fasttext.10runs.csv"
python fasttext.py --dataset 20newsgroups $common --seed $run --pickle-path ../pickles/20newsgroups.pickle	--learnable 0 	--lr 0.05	--nepochs 200	--pretrained
python fasttext.py --dataset 20newsgroups $common --seed $run --pickle-path ../pickles/20newsgroups.pickle	--learnable 0 	--lr 0.1	--nepochs 5	--pretrained --supervised
python fasttext.py --dataset 20newsgroups $common --seed $run --pickle-path ../pickles/20newsgroups.pickle	--learnable 300	--lr 0.5	--nepochs 200
python fasttext.py --dataset 20newsgroups $common --seed $run --pickle-path ../pickles/20newsgroups.pickle	--learnable 20 	--lr 0.05	--nepochs 200	--pretrained
python fasttext.py --dataset jrcall $common --seed $run --pickle-path ../pickles/jrcall.pickle			--learnable 0 	--lr 0.25	--nepochs 200	--pretrained
python fasttext.py --dataset jrcall $common --seed $run --pickle-path ../pickles/jrcall.pickle			--learnable 0 	--lr 0.25	--nepochs 100	--pretrained --supervised
python fasttext.py --dataset jrcall $common --seed $run --pickle-path ../pickles/jrcall.pickle			--learnable 300	--lr 0.25	--nepochs 200
python fasttext.py --dataset jrcall $common --seed $run --pickle-path ../pickles/jrcall.pickle			--learnable 300 --lr 0.25	--nepochs 200	--pretrained
python fasttext.py --dataset ohsumed $common --seed $run --pickle-path ../pickles/ohsumed.pickle		--learnable 0 	--lr 0.05	--nepochs 100	--pretrained
python fasttext.py --dataset ohsumed $common --seed $run --pickle-path ../pickles/ohsumed.pickle		--learnable 0 	--lr 0.5	--nepochs 5	--pretrained --supervised
python fasttext.py --dataset ohsumed $common --seed $run --pickle-path ../pickles/ohsumed.pickle		--learnable 200	--lr 0.5	--nepochs 200
python fasttext.py --dataset ohsumed $common --seed $run --pickle-path ../pickles/ohsumed.pickle		--learnable 23 	--lr 0.5	--nepochs 200	--pretrained
python fasttext.py --dataset rcv1 $common --seed $run --pickle-path ../pickles/rcv1.pickle			--learnable 0 	--lr 0.1	--nepochs 200	--pretrained
python fasttext.py --dataset rcv1 $common --seed $run --pickle-path ../pickles/rcv1.pickle			--learnable 0 	--lr 0.05	--nepochs 200	--pretrained --supervised
python fasttext.py --dataset rcv1 $common --seed $run --pickle-path ../pickles/rcv1.pickle			--learnable 50 	--lr 0.5	--nepochs 100
python fasttext.py --dataset rcv1 $common --seed $run --pickle-path ../pickles/rcv1.pickle			--learnable 101 --lr 0.1	--nepochs 200	--pretrained
python fasttext.py --dataset reuters21578 $common --seed $run --pickle-path ../pickles/reuters21578.pickle	--learnable 0 	--lr 0.5	--nepochs 100	--pretrained
python fasttext.py --dataset reuters21578 $common --seed $run --pickle-path ../pickles/reuters21578.pickle	--learnable 0 	--lr 0.5	--nepochs 200	--pretrained --supervised
python fasttext.py --dataset reuters21578 $common --seed $run --pickle-path ../pickles/reuters21578.pickle	--learnable 300	--lr 0.5	--nepochs 100
python fasttext.py --dataset reuters21578 $common --seed $run --pickle-path ../pickles/reuters21578.pickle	--learnable 115 --lr 0.5	--nepochs 100	--pretrained
python fasttext.py --dataset wipo-sl-sc $common --seed $run --pickle-path ../pickles/wipo-sl-sc.pickle		--learnable 0 	--lr 0.5	--nepochs 10	--pretrained
python fasttext.py --dataset wipo-sl-sc $common --seed $run --pickle-path ../pickles/wipo-sl-sc.pickle		--learnable 0 	--lr 0.5	--nepochs 10	--pretrained --supervised
python fasttext.py --dataset wipo-sl-sc $common --seed $run --pickle-path ../pickles/wipo-sl-sc.pickle		--learnable 300	--lr 0.5	--nepochs 10
python fasttext.py --dataset wipo-sl-sc $common --seed $run --pickle-path ../pickles/wipo-sl-sc.pickle		--learnable 300	--lr 0.5	--nepochs 10	--pretrained

done
