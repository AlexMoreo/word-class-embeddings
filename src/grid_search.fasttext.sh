#!/usr/bin/env bash
set -x

source activate torch

seed=1

for dataset in reuters21578 20newsgroups ohsumed rcv1 jrcall
do
    common="--dataset $dataset --pickle-path ../pickles/$dataset.pickle --dataset-dir ../fasttext/dataset --log-file ../log/fasttext.hyper.csv --seed $seed --validation"
    for nepochs in 5 100 200
    do
        for lr in 0.05 0.1 0.25 0.5
        do
            for learnable in 50 200
            do
                python fasttext.py $common --lr $lr --nepochs $nepochs --learnable $learnable
            done
        done
    done

done

for dataset in wipo-sl-mg ag-news amazon-review-full yahoo-answers yelp-review-full yelp-review-polarity amazon-review-polarity
do
    common="--dataset $dataset --pickle-path ../pickles/$dataset.pickle --dataset-dir ../fasttext/dataset --log-file ../log/fasttext.hyper.csv --seed $seed --validation"
    for nepochs in 5 100
    do
        for lr in 0.05 0.1 0.25 0.5
        do
            for learnable in 50 200
            do
                python fasttext.py $common --lr $lr --nepochs $nepochs --learnable $learnable
            done
        done
    done

done
