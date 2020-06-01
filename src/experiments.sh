#!/usr/bin/env bash
set -x

source activate torch

for seed in {0..0}
do
    stop="patience"
    if [ $seed -eq 0 ] ; then
        stop="epochs"
    fi

	for net in cnn # lstm # attn
	do
	    for dataset in reuters21578 # ohsumed 20newsgroups rcv1 jrc300 # imdb
	    do
          teach=1
	        if [ $dataset == "rcv1" ] ; then
	            teach=10
          fi
          echo "$teach"
          common="--dataset $dataset --net $net --pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv --seed $seed --stop $stop --test-each $teach"

          python3 main.py $common --glove
          python3 main.py $common --glove --supervised --sup-drop 0.5
	    done
	done
done
