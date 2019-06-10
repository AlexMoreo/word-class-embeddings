#!/usr/bin/env bash
set -x

source activate torch

seed=1

echo "sleeping..."
sleep 5s
echo "done"

for dataset in jrcall
do
    common="--dataset $dataset --log-file ../log/$dataset.hyper.csv --seed $seed"


    for net in attn
    do
        for hidden in 256 512 1024 2048
        do
            python main.py $common --net $net --hidden $hidden --pretrained glove
            sleep 60s
            python main.py $common --net $net --hidden $hidden --pretrained glove --supervised
            sleep 60s
        done
    done

done


for dataset in wipo-sl-mg ag-news amazon-review-full yahoo-answers yelp-review-full yelp-review-polarity amazon-review-polarity
do
    common="--dataset $dataset --log-file ../log/$dataset.hyper.csv --seed $seed"

    net=cnn
    for channels in 64 128 256 512
    do
#        for learnable in 50 200
#        do
#            python main.py $common --net $net --channels $channels --learnable $learnable
#        done
        python main.py $common --net $net --channels $channels --pretrained glove
        sleep 60s
        python main.py $common --net $net --channels $channels --pretrained glove --supervised
        sleep 60s
    done

    for net in lstm attn
    do
        for hidden in 256 512 1024 2048
        do
#            for learnable in 50 200
#            do
#                python main.py $common --net $net --hideen $hidden --learnable $learnable
#            done
            python main.py $common --net $net --hidden $hidden --pretrained glove
            sleep 60s
            python main.py $common --net $net --hidden $hidden --pretrained glove --supervised
            sleep 60s
        done
    done

done


