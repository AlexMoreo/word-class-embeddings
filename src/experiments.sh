#!/usr/bin/env bash
set -x

source activate torch

for seed in {0..5}
do
    stop="patience"
    if [ $seed -eq 0 ] ; then
        stop="epochs"
    fi

	for net in cnn lstm # attn
	do
	    for dataset in reuters21578 ohsumed 20newsgroups rcv1 jrc300 # imdb
	    do
           teach=1
	        if [ $dataset == "rcv1" ] ; then
	            teach=10
            fi
            echo "$teach"
            common="--dataset $dataset --net $net --pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv --seed $seed --stop $stop --test-each $teach"

            python main.py $common --glove
            python main.py $common --glove --supervised
            python main.py $common --glove --supervised --sup-drop 0.5
            python main.py $common --learnable 300
	    done

	done
done

#for dataset in 20newsgroups reuters21578 ohsumed rcv1 jrc300
#do
#	python baselines.py --dataset $dataset --pickle-path ../pickles/$dataset.pickle --log-file ../log/baselines.csv
#done
#
#for dataset in 20newsgroups reuters21578 ohsumed rcv1 jrc300
#do
#	python baselines.py --dataset $dataset --pickle-path ../pickles/$dataset.pickle --log-file ../log/baselines.csv --mode supervised
#done



#
#
#
#
#
#
#
#
#
#for net in cnn lstm #attn
#do
#    for dataset in reuters21578 ohsumed 20newsgroups rcv1 # imdb
#    do
#        python main.py --dataset $dataset --net $net --glove 					--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
#        python main.py --dataset $dataset --net $net --glove --supervised			--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
#        python main.py --dataset $dataset --net $net --glove --supervised --sup-drop 0.5	--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
###        python main.py --dataset $dataset --net $net --glove --supervised --sentiment 		--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
##        python main.py --dataset $dataset --net $net --glove --word-drop 0.05			--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
##        python main.py --dataset $dataset --net $net --glove --supervised --predict-all 	--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
##        python main.py --dataset $dataset --net $net --glove --supervised --predict-missing 	--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
##        python main.py --dataset $dataset --net $net --glove --supervised --word-drop 0.05	--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
##        python main.py --dataset $dataset --net $net --glove --supervised --learnable 100 	--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
##        python main.py --dataset $dataset --net $net --learnable 300 				--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
##        python main.py --dataset $dataset --net $net --supervised 				--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
#    done
#done
#
#for net in cnn lstm
#do
#    for dataset in jrcall
#    do
#        python main.py --dataset $dataset --net $net --glove 					--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
#        python main.py --dataset $dataset --net $net --glove --supervised --max-label-space 100	--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
#        python main.py --dataset $dataset --net $net --glove --supervised --max-label-space 300	--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
#        python main.py --dataset $dataset --net $net --glove --supervised --sup-drop 0.5 --max-label-space 300	--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
#        python main.py --dataset $dataset --net $net --glove --supervised --sup-drop 0.5 --max-label-space 300 --pca-whitening --pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
#
#    done
#done
#
#for net in attn
#do
#    for dataset in reuters21578 ohsumed 20newsgroups rcv1 # imdb
#    do
#        python main.py --dataset $dataset --net $net --glove 					--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
#        python main.py --dataset $dataset --net $net --glove --supervised			--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
#        python main.py --dataset $dataset --net $net --glove --supervised --sup-drop 0.5	--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
#    done
#done
#
#for net in attn
#do
#    for dataset in jrcall
#    do
#        python main.py --dataset $dataset --net $net --glove 					--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
#        python main.py --dataset $dataset --net $net --glove --supervised --max-label-space 100	--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
#        python main.py --dataset $dataset --net $net --glove --supervised --max-label-space 300	--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
#        python main.py --dataset $dataset --net $net --glove --supervised --sup-drop 0.5 --max-label-space 300	--pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
#        python main.py --dataset $dataset --net $net --glove --supervised --sup-drop 0.5 --max-label-space 300 --pca-whitening --pickle-path ../pickles/$dataset.pickle --log-file ../log/$dataset.csv
#
#    done
#done
