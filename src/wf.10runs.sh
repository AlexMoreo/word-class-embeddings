#!/usr/bin/env bash
set -x

source activate torch

common="--sup-drop 0 --nozscore --supervised-method wp"

log="--log-file ../log/10runs.wp.csv"
for run in {1..10}
do

dataset=20newsgroups
#python main.py $common $log  --dataset $dataset	--net cnn	--channels 128	--pretrained glove	--supervised --seed $run
#python main.py $common $log  --dataset $dataset	--net cnn	--channels 128	--pretrained glove	--supervised	--tunable --seed $run

dataset=ohsumed
#python main.py $common $log  --dataset $dataset	--net cnn	--channels 256	--pretrained glove	--supervised --seed $run
#python main.py $common $log  --dataset $dataset	--net cnn	--channels 256	--pretrained glove	--supervised	--tunable --seed $run

dataset=reuters21578
#python main.py $common $log  --dataset $dataset	--net cnn	--channels 512	--pretrained glove	--supervised	 --seed $run
#python main.py $common $log  --dataset $dataset	--net cnn	--channels 512	--pretrained glove	--supervised	--tunable	 --seed $run

dataset=rcv1
#python main.py $common $log  --dataset $dataset	--net cnn	--channels 256	--pretrained glove	--supervised --seed $run 
#python main.py $common $log  --dataset $dataset	--net cnn	--channels 256	--pretrained glove	--supervised	--tunable --seed $run 





dataset=wipo-sl-sc
#python main.py $common $log  --dataset $dataset	--net cnn	--channels 128	--pretrained glove	--supervised --seed $run --max-epoch-length 300 
python main.py $common $log  --dataset $dataset	--net cnn  --channels 128	--pretrained glove	--supervised --tunable --seed $run --max-epoch-length 300 --max-label-space -1

dataset=jrcall
#python main.py $common $log  --dataset $dataset	--net cnn	--channels 256	--pretrained glove	--supervised    --seed $run
#python main.py $common $log  --dataset $dataset	--net cnn	--channels 256	--pretrained glove	--supervised    --tunable --seed $run --max-label-space -1

done

