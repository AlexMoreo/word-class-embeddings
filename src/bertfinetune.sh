python main_bertfinetune.py --force --batch-size 16 --dataset 20newsgroups --log-file ../log/20newsgroups.plot.cw.csv --seed 0 --test-each 1 --plotmode --nepochs 100
python main_bertfinetune.py --force --batch-size 16 --dataset ohsumed --log-file ../log/ohsumed.plot.cw.csv --seed 0 --test-each 1 --plotmode --nepochs 100
python main_bertfinetune.py --force --batch-size 16 --dataset jrcall --log-file ../log/jrcall.plot.cw.csv --seed 0 --test-each 1 --plotmode --nepochs 100
python main_bertfinetune.py --force --batch-size 16 --dataset reuters21578 --log-file ../log/reuters21578.plot.cw.csv --seed 0 --test-each 1 --plotmode --nepochs 100
python main_bertfinetune.py --force --batch-size 16 --dataset rcv1 --log-file ../log/rcv1.plot.cw.csv --seed 0 --test-each 10 --plotmode --nepochs 100
python main_bertfinetune.py --force --batch-size 16 --dataset wipo-sl-sc --log-file ../log/wipo-sl-sc.plot.cw.csv --seed 0 --test-each 10 --plotmode --nepochs 100 --max-epoch-length 300
