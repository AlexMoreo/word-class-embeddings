#!/usr/bin/env bash
set -x

source activate torch

seed=1

log="--log-file ../log/fasttext.hyper.pcaend.csv"
for dataset in jrcall #reuters21578 20newsgroups ohsumed rcv1 jrcall
do
    if [ "$dataset" == "reuters21578" ]; then
       ncats=115
    elif [ "$dataset" == "20newsgroups" ]; then
       ncats=20
    elif [ "$dataset" == "ohsumed" ]; then
       ncats=23
    elif [ "$dataset" == "rcv1" ]; then
       ncats=101
    elif [ "$dataset" == "jrcall" ]; then
       ncats=300
    fi
    echo "$dataset -> $ncats categories"
    common="--dataset $dataset --pickle-path ../pickles/$dataset.pickle --dataset-dir ../fasttext/dataset $log --seed $seed"
    for nepochs in 5 100 200
    do
        for lr in 0.05 0.1 0.25 0.5
        do
            #for learnable in 50 200 300
            #do
            #    python fasttext.py $common --lr $lr --nepochs $nepochs --learnable $learnable
            #done
	    #python fasttext.py $common --lr $lr --nepochs $nepochs --learnable 0 --pretrained
	    python fasttext.py $common --lr $lr --nepochs $nepochs --learnable 0 --pretrained --supervised
#	    python fasttext.py $common --lr $lr --nepochs $nepochs --learnable $ncats --pretrained
        done
    done
done

ncats=300
for nepochs in 5 10
do
    for dataset in wipo-sl-sc # wipo-sl-sc ag-news amazon-review-full yahoo-answers yelp-review-full yelp-review-polarity amazon-review-polarity
    do
	common="--dataset $dataset --pickle-path ../pickles/$dataset.pickle --dataset-dir ../fasttext/dataset $log --seed $seed"
        for lr in 0.05 0.1 0.25 0.5
        do
            #for learnable in 50 200 300
            #do
            #    python fasttext.py $common --lr $lr --nepochs $nepochs --learnable $learnable
            #done
	    #python fasttext.py $common --lr $lr --nepochs $nepochs --learnable 0 --pretrained
	    python fasttext.py $common --lr $lr --nepochs $nepochs --learnable 0 --pretrained --supervised
#	    python fasttext.py $common --lr $lr --nepochs $nepochs --learnable $ncats --pretrained
        done
    done

done
