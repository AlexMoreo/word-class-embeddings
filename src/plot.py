import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from data.dataset import Dataset
from util.csv_log import CSVLog
from util.file import create_if_not_exist
import pandas as pd


def identity(x): return x

def plot(csv_file_path, dataset, index_by='epoch', plotdir='./plots', smooth_tr_loss=True, log_y=True, baselines=None,
         show_stop=False,
         nets=['cnn', 'lstm', 'attn'],
         methods=None,
         measures=None,
         preferred_names=identity,
         xmax=None):

    create_if_not_exist(plotdir)
    csv = CSVLog(csv_file_path).df

    if methods is None:
        methods = sorted(csv.method.unique())
    if measures is None:
        measures = sorted(csv.measure.unique())

    sns.set_style("darkgrid")
    for net in nets:
        for measure in measures:
            if measure=='early-stop': continue
            if measure.startswith('final-te'): continue
            measure_name = measure
            fig, ax = plt.subplots()

            first_lims = True
            min_y, max_y = None, None
            baseline_data = []
            if baselines is not None:
                from_measure = baselines['measure']==measure
                from_dataset = baselines['dataset']==dataset
                results = baselines[from_measure & from_dataset]
                if not results.empty:
                    baseline_methods = np.unique(baselines.method)
                    for baseline_method in baseline_methods:
                        if baseline_method=='SVM-S': continue
                        from_method = results['method'] == baseline_method
                        baseline_results = results[from_method]
                        score=baseline_results.value.values[0]
                        baseline_data.append((baseline_method,score))

            added_baseline = False
            for method in methods:
                if not method.startswith(net): continue

                from_method = csv['method']==method
                from_measure = csv['measure']==measure
                from_dataset = csv['dataset']==dataset
                results = csv[from_method & from_measure & from_dataset]
                results = results.pivot_table(index=['run',index_by], values='value')

                long_run = 0
                if long_run not in results.index: continue
                results = results.loc[long_run] # only the first run

                if results.empty: continue

                xs = results.index.values.astype(int)
                ys = results.values.flatten()

                if log_y and 'loss' in measure:
                    ys = np.log(ys)
                    measure_name = f'log({measure})'

                stop_val = None
                if show_stop:
                    from_earystop = csv['measure'] == 'early-stop'
                    from_firstrun = csv['run'] == 0
                    stop_point = csv[from_method & from_earystop & from_dataset & from_firstrun][index_by].values
                    if len(stop_point)>1:
                        print('more than one stop-points')
                        stop_val = None
                    else:
                        if len(stop_point)>0:
                            stop_point = stop_point[0]
                            for i,xsi in enumerate(xs):
                                if xsi >= stop_point: break
                            stop_val=ys[i]

                # ys = smooth(ys)
                p = ax.plot(xs, ys, '-', label=preferred_names(method))
                if stop_val:
                    last_color = p[-1].get_color()
                    ax.plot(stop_point, stop_val, 'o', color=last_color, markersize=10)
                if baseline_data and not added_baseline:
                    for b_name, b_score in baseline_data:
                        ax.axhline(b_score, linestyle='--', color='gray')
                        plt.text(max(xs), b_score, b_name,
                                 horizontalalignment='right',
                                 verticalalignment='bottom',
                                 multialignment='right')
                        added_baseline=True

                ax.legend()

                if first_lims:
                    min_y, max_y = ys.min(), ys.max()
                    first_lims = False
                else:
                    min_y, max_y = np.min((min_y, ys.min())), np.max((max_y, ys.max()))

            # index_by = 'Wall-clock' if index_by == 'timelapse' else index_by
            ax.set(xlabel=index_by, ylabel=measure_name)#, title=f'Classification Performance in {dataset.title()}')

            if first_lims is False:
                interval = (max_y-min_y)*0.1
                if interval>0:
                    ax.set_ylim((min_y-interval, max_y+interval))
                    if xmax is not None: ax.set_xlim((0, xmax))
                    # if not os.path.exists(f'{plotdir}/{net}'): os.makedirs(f'{plotdir}/{net}')
                    # fig.savefig(f'{plotdir}/{net}/{measure}_by_{index_by}.png', bbox_inches='tight')
                    fig.savefig(f'{plotdir}-{net}-{measure}_by_{index_by}.png', bbox_inches='tight')
                    plt.close('all')


def smooth(x, window=11):
    if len(x)<window: return x
    from scipy.signal import savgol_filter
    return savgol_filter(x, window, 3)

def preferred_names(method_name):
    method_name = method_name.replace('cnn', 'CNN').replace('lstm', 'LSTM').replace('attn', 'ATTN')
    method_name = method_name.replace('-ch256', '').replace('-h512', '')
    method_name = method_name.replace('-d0.5-dotn', '').replace('-d0.2-dotn', '')
    method_name = method_name.replace('-glove', '-GloVe').replace('-word2vec','-Word2Vec').replace('-supervised', '+Sup').replace('-learn200','+Rand')
    return method_name

def plot_training_trends():
    print('[plot:training_trends]')
    outpath = '../latex/plots/training_trend_plots'
    common_methods=['cnn-glove-ch256', 'cnn-learn200-ch256',
               'lstm-glove-h512', 'lstm-learn200-h512',
               'attn-glove-h512', 'attn-learn200-h512']

    for dataset in ['jrcall', 'wipo-sl-sc', 'rcv1', 'ohsumed', '20newsgroups', 'reuters21578']:
        if dataset=='wipo-sl-sc':
            methods=common_methods+['cnn-glove-supervised-d0.2-dotn-ch256','lstm-glove-supervised-d0.2-dotn-h512', 'attn-glove-supervised-d0.2-dotn-h512']
        else:
            methods = common_methods + ['cnn-glove-supervised-d0.5-dotn-ch256', 'lstm-glove-supervised-d0.5-dotn-h512', 'attn-glove-supervised-d0.5-dotn-h512']
        csvpath = f'../log/{dataset}.plot.csv'
        print(f'plotting {dataset}')
        plot(csvpath, dataset, plotdir=f'{outpath}/{dataset}', methods=methods, preferred_names=preferred_names, xmax=100)


def plot_glove_vs_word2vec():
    print('[plot:glove_vs_word2vec]')
    outpath = '../latex/plots/glove_vs_word2vec'
    methods = ['cnn-glove-ch256', 'cnn-glove-supervised-d0.5-dotn-ch256', 'cnn-word2vec-ch256', 'cnn-word2vec-supervised-d0.5-dotn-ch256',
               'lstm-glove-h512', 'lstm-glove-supervised-d0.5-dotn-h512', 'lstm-word2vec-h512', 'lstm-word2vec-supervised-d0.5-dotn-h512',
               'attn-glove-h512', 'attn-glove-supervised-d0.5-dotn-h512', 'attn-word2vec-h512', 'attn-word2vec-supervised-d0.5-dotn-h512']
    for dataset in ['jrcall', 'rcv1', 'ohsumed', '20newsgroups', 'reuters21578']:
        csvpath = f'../log/{dataset}.plot.csv'
        print(f'plotting {dataset}')
        plot(csvpath, dataset, plotdir=f'{outpath}/{dataset}', methods=methods, preferred_names=preferred_names, xmax=100, measures=['te-macro-F1'])

    methods = ['cnn-glove-ch256', 'cnn-glove-supervised-d0.2-dotn-ch256', 'cnn-word2vec-ch256','cnn-word2vec-supervised-d0.2-dotn-ch256',
               'lstm-glove-h512', 'lstm-glove-supervised-d0.2-dotn-h512', 'lstm-word2vec-h512','lstm-word2vec-supervised-d0.2-dotn-h512',
               'attn-glove-h512', 'attn-glove-supervised-d0.2-dotn-h512', 'attn-word2vec-h512','attn-word2vec-supervised-d0.2-dotn-h512']
    for dataset in ['wipo-sl-sc']:
        csvpath = f'../log/{dataset}.plot.csv'
        print(f'plotting {dataset}')
        plot(csvpath, dataset, plotdir=f'{outpath}/{dataset}', methods=methods, preferred_names=preferred_names, xmax=100, measures=['te-macro-F1'])

def plot_regularization():
    print('[plot:regularization]')
    outpath = '../latex/plots/regularization'
    methods = ['cnn-glove-supervised-d0.0-dotn-ch256', 'cnn-glove-supervised-d0.2-dotn-ch256', 'cnn-glove-supervised-d0.5-dotn-ch256', 'cnn-glove-supervised-d0.8-dotn-ch256',
               'lstm-glove-supervised-d0.0-dotn-h512', 'lstm-glove-supervised-d0.2-dotn-h512', 'lstm-glove-supervised-d0.5-dotn-h512', 'lstm-glove-supervised-d0.8-dotn-h512',
               'attn-glove-supervised-d0.0-dotn-h512', 'attn-glove-supervised-d0.2-dotn-h512', 'attn-glove-supervised-d0.5-dotn-h512', 'attn-glove-supervised-d0.8-dotn-h512']

    def preferred_names_reg(method_name):
        method_name = method_name.replace('cnn', 'CNN').replace('lstm', 'LSTM').replace('attn', 'ATTN')
        method_name = method_name.replace('-ch256', '').replace('-h512', '')
        method_name = method_name.replace('-d0.5-dotn', '(d=0.5)').replace('-d0.2-dotn', '(d=0.2)').replace('-d0.8-dotn', '(d=0.8)').replace('-d0.0-dotn', '(d=0)')
        method_name = method_name.replace('-glove', '-GloVe').replace('-supervised', '+Sup').replace('-learn200','+Rand')
        return method_name

    for dataset in ['wipo-sl-sc','ohsumed', 'jrcall', 'rcv1', 'reuters21578', '20newsgroups']:
        csvpath = f'../log/{dataset}.plot.csv'
        print(f'plotting {dataset}')
        plot(csvpath, dataset, plotdir=f'{outpath}/{dataset}', methods=methods, preferred_names=preferred_names_reg, xmax=100)

def plot_supervised_functions():
    print('[plot:supervised_functions]')
    outpath = '../latex/plots/supervised_functions'
    functions = ['dotn', 'pmi', 'ig', 'chi2', 'gss'] #'pnig',
    nets = ['cnn-glove-supervised','lstm-glove-supervised','attn-glove-supervised']

    def preferred_names(method_name):
        method_name = method_name.replace('cnn', 'CNN').replace('lstm', 'LSTM').replace('attn', 'ATTN')
        method_name = method_name.replace('-ch256', '').replace('-h512', '')
        method_name = method_name.replace('-d0.5', '').replace('-d0.2', '')
        method_name = method_name.replace('-glove', '').replace('-supervised','')
        method_name = method_name.replace('dotn','Dot').replace('pmi','PMI').replace('pnig','AsymInfoGain')\
            .replace('ig','InfoGain').replace('chi2','$\chi^2$').replace('gss','GSS')
        return method_name

    for dataset in ['wipo-sl-sc', 'jrcall', 'rcv1', 'ohsumed', '20newsgroups', 'reuters21578']:
        d=0.5
        if dataset=='wipo-sl-sc':
            d=0.2

        methods = [f'{net}-d{d}-{function}-{("ch256" if net.startswith("cnn") else "h512")}' for net in nets for function in functions]
        csvpath = f'../log/{dataset}.plot.csv'
        print(f'plotting {dataset}')
        plot(csvpath, dataset, plotdir=f'{outpath}/{dataset}', methods=methods, preferred_names=preferred_names, xmax=100, measures=['te-macro-F1'])

def plot_oov_predictions():
    print('[plot:oov_predictions]')
    outpath = '../latex/plots/oov_predictions'
    functions = ['-miss','-all', '', '-miss3', '-all3']
    nets = ['cnn-glove-supervised', 'lstm-glove-supervised', 'attn-glove-supervised']

    def preferred_names(method_name):
        method_name = method_name.replace('cnn', 'CNN').replace('lstm', 'LSTM').replace('attn', 'ATTN')
        method_name = method_name.replace('-ch256', '').replace('-h512', '')
        method_name = method_name.replace('-d0.5', '').replace('-d0.2', '')
        method_name = method_name.replace('-glove', '').replace('-supervised', '')
        method_name = method_name.replace('-dotn', '').replace('-miss', '-Missing').replace('-all', '-All')
        return method_name

    for dataset in ['rcv1']:
        d = 0.5
        methods = [f'{net}{function}-d{d}-dotn-{("ch256" if net.startswith("cnn") else "h512")}' for net in nets for function in functions]
        csvpath = f'../log/{dataset}.plot.csv'
        print(f'plotting {dataset}')
        plot(csvpath, dataset, plotdir=f'{outpath}/{dataset}', methods=methods, preferred_names=preferred_names, xmax=100, measures=['te-macro-F1', 'te-micro-F1'])


if __name__ == '__main__':
    sns.set_context("talk")
    # plot_training_trends()
    # plot_glove_vs_word2vec()
    # plot_regularization()
    plot_supervised_functions()
    # plot_oov_predictions()