import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from data.dataset import Dataset
from util.csv_log import CSVLog
from util.file import create_if_not_exist
import pandas as pd


def plot(csv_file_path, dataset, index_by='epoch', plotdir='./plots', smooth_tr_loss=True, log_y=True, baselines=None):
    create_if_not_exist(plotdir)

    csv = CSVLog(csv_file_path).df

    methods = sorted(csv.method.unique())
    measures = sorted(csv.measure.unique())

    for net in ['cnn', 'lstm', 'attn']:
        for measure in measures:
            if measure=='early-stop': continue
            if measure.startswith('final-te'): continue
            measure_name = measure
            fig, ax = plt.subplots()
            sns.set_style("darkgrid")
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


                long_run = 1
                if long_run not in results.index: continue
                results = results.loc[long_run] # only the first run
                # results = results.pivot_table(index=[index_by], values='value', aggfunc=lambda x:(np.mean(x),np.std(x)))

                if results.empty: continue

                xs = results.index.values.astype(int)
                ys = results.values.flatten()
                # ys=np.asarray(results.values.flatten().tolist())
                # ys_mean, ys_std = ys[:,0], ys[:,1]

                if log_y and 'loss' in measure:
                    ys = np.log(ys)
                    measure_name = f'log({measure})'

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
                    else:
                        stop_val=None

                # if smooth_tr_loss:# and measure=='tr_loss':
                #     ys = smooth(ys)

                # p = ax.plot(xs, ys_mean, '-', label=method)
                # last_color = p[-1].get_color()
                # ax.fill_between(xs, ys_mean - ys_std, ys_mean + ys_std, color=last_color, alpha=0.5)

                p = ax.plot(xs, ys, '-', label=method)
                if stop_val:
                    last_color = p[-1].get_color()
                    ax.plot(stop_point, stop_val, 'o', color=last_color, markersize=10)
                if baseline_data and not added_baseline:
                    for b_name, b_score in baseline_data:
                        xs_=[min(xs),max(xs)]
                        ys_=[b_score]*2
                        # ax.plot(xs_,ys_,'--',label=b_name)
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
            ax.set(xlabel=index_by, ylabel=measure_name, title=f'Classification Performance in {dataset.title()}')

            if first_lims is False:
                interval = (max_y-min_y)*0.1
                if interval>0:
                    ax.set_ylim((min_y-interval, max_y+interval))
                    if not os.path.exists(f'{plotdir}/{net}'): os.makedirs(f'{plotdir}/{net}')
                    fig.savefig(f'{plotdir}/{net}/{measure}_by_{index_by}.png')
                    plt.close('all')


def evaluation(csv_file_path, dataset, numerical, measures=['te-macro-F1', 'te-micro-F1']):
    if not os.path.exists(csv_file_path): return
    csv = CSVLog(csv_file_path).df

    methods = sorted(csv.method.unique())

    for net in {'cnn', 'lstm', 'attn'}:
        for measure in measures:

            for method in methods:
                if not method.startswith(net): continue
                # if not method.endswith('supervised'): continue

                from_method = csv['method'] == method
                from_measure = csv['measure'] == measure
                from_dataset = csv['dataset'] == dataset
                results = csv[from_method & from_measure & from_dataset]
                results = results.pivot_table(index=['run', index_by], values='value')

                if results.empty: continue

                from_earystop = csv['measure'] == 'early-stop'
                stop_selection = csv[from_method & from_earystop & from_dataset]
                stop_point = stop_selection[index_by].values
                stop_times = stop_selection['timelapse'].values
                # print(f'number of points: {len(stop_point)}')
                values=[]
                for p,stop_point_i in enumerate(stop_point):
                    xs = results.loc[p].index.values.flatten()
                    ys = results.loc[p].values.flatten()
                    for i, xsi in enumerate(xs):
                        if xsi >= stop_point_i: break
                    stop_val = ys[i]
                    values.append(stop_val)

                # print(f'{dataset} {method} {measure}')

                if values:
                    mean = np.mean(values) if len(values)>1 else values[0]
                    std = np.std(values) if len(values)>1 else 0
                    time_mean = np.mean(stop_times) if len(stop_times)>1 else stop_times[0]
                    time_std  = np.std(stop_times) if len(stop_times)>1 else 0
                    # print(f'{dataset} {method} {measure} = {mean:.3f}(+-{std:.3f})')
                    numerical.add_row(dataset=dataset, method=method, measure=measure, mean=mean, std=std, runs=len(values), timelapse=time_mean)#f"{time_mean:.1f} (+-{time_std:.3f})")


def smooth(x, window=11):
    if len(x)<window: return x
    from scipy.signal import savgol_filter
    return savgol_filter(x, window, 3)

if __name__ == '__main__':
    baselines = CSVLog('../log/baselines.csv').df
    numerical = CSVLog('../results/numerical.csv', ['dataset', 'method', 'runs', 'measure', 'mean', 'std', 'timelapse'], overwrite=True)
    for index_by in ['epoch']:
        # for dataset in Dataset.dataset_available:
        for dataset in {'jrcall'}:#, 'amazon-review-full', 'amazon-review-polarity', 'yahoo-answers', 'yelp-review-full', 'yelp-review-polarity'}:
            # if dataset in ['imdb']: continue
            # csvpath = f'../log/{dataset}.hyper.csv'
            csvpath = f'../log/jrcall.tmp.csv'
            print(f'plotting {dataset}')
            plot(csvpath, dataset, index_by, plotdir=f'../plots/{dataset}', baselines=baselines)
            # evaluation(csvpath, dataset, numerical)


    baselines = baselines.rename(index=str, columns={"value": "mean"})

    baselines = baselines[baselines['measure'] != 'te-accuracy']
    baselines['std']=0
    baselines['runs'] = 1
    print(baselines)

    # numerical.df['timelapse']=-1
    print(numerical.df)

    merged = pd.concat([baselines, numerical.df])
    print(merged)
    pt = merged.pivot_table(index=['dataset','method'], columns=['measure'], values=['mean','std', 'timelapse'])
    print('='*80)
    print(str(pt).replace('0.000000', '-'*8))
