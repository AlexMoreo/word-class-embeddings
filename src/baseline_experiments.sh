#!/usr/bin/env bash
#set -x

source activate torch

for dataset in reuters21578 ohsumed rcv1 jrcall 20newsgroups wipo-sl-sc
do
#	python svm_baselines.py --dataset $dataset --pickle-path ../pickles/$dataset.pickle --log-file ../log/baselines.csv --optimc
#	python svm_baselines.py --dataset $dataset --pickle-path ../pickles/$dataset.pickle --log-file ../log/baselines_no_model_selection.csv --mode tfidf
#	python svm_baselines.py --dataset $dataset --pickle-path ../pickles/$dataset.pickle --log-file ../log/baselines_no_model_selection.csv --mode supervised
	python svm_baselines.py --dataset $dataset --pickle-path ../pickles/$dataset.pickle --log-file ../log/baselines_no_model_selection.csv --mode tfidf --learner lr
	python svm_baselines.py --dataset $dataset --pickle-path ../pickles/$dataset.pickle --log-file ../log/baselines_no_model_selection.csv --mode supervised  --learner lr
done

#for dataset in reuters21578 20newsgroups
#do
#	python baselines.py --dataset $dataset --pickle-path ../pickles/$dataset.pickle --log-file ../log/baselines.tmp_.csv
#done

