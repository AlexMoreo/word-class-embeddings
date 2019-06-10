#!/usr/bin/env bash
#set -x

source activate torch

#for dataset in jrcall #reuters21578 ohsumed rcv1 jrc300
#do
#	python baselines.py --dataset $dataset --pickle-path ../pickles/$dataset.pickle --log-file ../log/baselines.csv --optimc
#done

for dataset in reuters21578 20newsgroups
do
	python baselines.py --dataset $dataset --pickle-path ../pickles/$dataset.pickle --log-file ../log/baselines.tmp_.csv
done

