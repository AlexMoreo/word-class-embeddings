import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

svm_file = '../log/baselines.csv'


df = pd.read_csv(svm_file, sep='\t')

nc = {'reuters21578':115, '20newsgroups':20, 'rcv1':101, 'jrcall':2600, 'ohsumed':23, 'jrc300':300}

for method in ['SVM-tfidf', 'SVM-S']:
    dfs = df[df.method==method]
    dfs = dfs[dfs.measure == 'te-macro-F1']
    datasets,values = dfs.dataset.values, dfs.timelapse.values
    # pos = np.arange(len(datasets))
    # pos = [nc[d] for d in datasets]

    for d,v in zip(datasets, values):
        print(f'{method}\t{d}\t{v:.1f}')

    # print(datasets, values)
    # print(dfs)

    # pos = np.log(pos)
    # values = np.log(values)

    # plt.bar(pos, values, align='center', alpha=0.5)
    # plt.xticks(pos, datasets)
    # plt.ylabel('Time (s)')
    # plt.title('Time')
    #
    # plt.show()

for dataset in nc.keys():
    if dataset=='jrc300': continue
    net_file = f'../log/{dataset}.plot.csv' # mejor con los 10runs...
    df = pd.read_csv(net_file, sep='\t')
    dfs = df[df.measure=='early-stop']
    dfs = pd.pivot_table(dfs, index=['method'], values=['timelapse'], aggfunc=lambda x:f'{np.mean(x):.1f} {np.std(x)}')
    print(dfs)

