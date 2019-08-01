#!/usr/bin/env bash
set -x

source activate torch

seed=1


#for dataset in wipo-sl-sc
#do
#    for emb in 200
#    do
#
#    common="--dataset $dataset --log-file ../log/$dataset.hyper.csv --seed 1 --test-each 0 --patience 10 --fullpickle ../pickles/$dataset-index.pickle --max-epoch-length 300"
#
#    net=cnn
#    for channels in 128 256 512
#    do
#        python main.py $common --net $net --channels $channels --learnable $emb
#    done
#
#
#    for net in lstm attn
#    do
#        for hidden in 256 512 1024 #2048
#        do
#            python main.py $common --net $net --hidden $hidden --learnable $emb
#        done
#    done
#
#    done
#
#done

