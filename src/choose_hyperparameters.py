import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from data.dataset import Dataset
from util.csv_log import CSVLog
from util.file import create_if_not_exist
import pandas as pd
from util.file import list_files

def process_method_name(method):
    method = method.replace('glove-sdrop', 'glove')
    method_parts = method.split('-')
    net = method_parts[0]
    method_variant = []
    params = ''
    for part in method_parts[1:]:
        if part=='glove':
            method_variant.append(part)
        elif part=='supervised':
            method_variant.append(part)
        elif part == 'sdrop':
            method_variant.append(part)
        else:
            params += part
    method_variant = '-'.join(method_variant)
    if not params:
        params = '-def'
    return net, method_variant, params

def bold_best(dataset_sel):
    values = dataset_sel.te_macro_f1.values
    best_macro_f1 = np.argmax([float(v) for v in values])
    best_macro_f1_str = values[best_macro_f1]
    best.df.loc[(best.df.dataset == dataset) & (best.df.te_macro_f1 == best_macro_f1_str), 'te_macro_f1'] = f'*{best_macro_f1_str}*'

    values = dataset_sel.te_micro_f1.values
    best_micro_f1 = np.argmax([float(v) for v in values])
    best_micro_f1_str = values[best_micro_f1]
    best.df.loc[(best.df.dataset == dataset) & (best.df.te_micro_f1 == best_micro_f1_str), 'te_micro_f1'] = f'*{best_micro_f1_str}*'

if __name__ == '__main__':
    import sys

    table = CSVLog('../results/hyper.all.csv', ['dataset', 'net','variant','params', 'va_macro_f1', 'te_macro_f1', 'te_micro_f1'], overwrite=True)

    dataset_results = list_files('../log')
    dataset_results = [file for file in dataset_results if file.endswith('hyper.csv')]

    merge = []
    for dataset in dataset_results:
        csvpath = f'../log/{dataset}'
        print(csvpath)
        df = pd.read_csv(csvpath, sep='\t')
        merge.append(df)

    df = pd.concat(merge)
    assert len(np.unique(df.run))==1, 'error, more than one run'

    print(df)
    datasets = np.unique(df.dataset)
    nets=set()
    methods=set()

    for dataset in datasets:
        df_data = df[df.dataset==dataset]
        for method in np.unique(df_data.method):
            df_data_method = df_data[df_data.method==method]
            stop_point = df_data_method[df_data_method.measure == 'early-stop']
            if not stop_point.empty:
                stop_point = stop_point.epoch.values[0]
                stop_va_macro_f1   = df_data_method[(df_data_method.measure == 'va-macro-F1') & (df_data_method.epoch == stop_point)].value.values[0]
                macro_f1_te = df_data_method[(df_data_method.measure == 'final-te-macro-F1')].value.values[0]
                micro_f1_te = df_data_method[(df_data_method.measure == 'final-te-micro-F1')].value.values[0]
                net,method,params = process_method_name(method)
                nets.add(net)
                methods.add(method)
                print(f'{dataset} {method} {stop_point} {stop_va_macro_f1:.3f}: {macro_f1_te:.3f} {micro_f1_te:.3f}')
                table.add_row(dataset=dataset, net=net, variant=method, params=params, va_macro_f1=stop_va_macro_f1,
                              te_macro_f1=macro_f1_te, te_micro_f1=micro_f1_te)

    best = CSVLog('../results/hyper.best.csv', ['dataset', 'net', 'variant', 'params', 'te_macro_f1', 'te_micro_f1'], overwrite=True)
    baselines = CSVLog('../log/baselines.csv').df

    pv = table.df.pivot_table(index=['dataset', 'net', 'variant', 'params'], values=['va_macro_f1', 'te_macro_f1', 'te_micro_f1'])
    print(pv)
    for dataset in datasets:
        if dataset not in pv.index: continue
        sel1 = pv.loc[dataset]
        if dataset in baselines.dataset.unique():
            presel = baselines[(baselines.dataset==dataset) & (baselines.method=='SVM-tfidf')]
            base_macro_f1 = presel[baselines.measure=='te-macro-F1'].value.values[0]
            base_micro_f1 = presel[baselines.measure=='te-micro-F1'].value.values[0]
            best.add_row(dataset=dataset, net='svm', variant='tfidf', params='-', te_macro_f1=f'{base_macro_f1:.3f}', te_micro_f1=f'{base_micro_f1:.3f}')

        for net in nets:
            if net not in sel1.index: continue
            sel2 = sel1.loc[net]
            for var in methods:
                if var not in sel2.index: continue
                print(dataset, net, var)
                sel3 = sel2.loc[var]
                print(sel3)
                best_val_pos = np.argmax(sel3.va_macro_f1.values)
                best_param  = sel3.index.values[best_val_pos]
                chosen_macro_f1 = f'{sel3.te_macro_f1.values[best_val_pos]:.3f}'
                chosen_micro_f1 = f'{sel3.te_micro_f1.values[best_val_pos]:.3f}'
                best.add_row(dataset=dataset, net=net, variant=var, params=best_param, te_macro_f1=chosen_macro_f1, te_micro_f1=chosen_micro_f1)

    # bold-face best result for each dataset
    for dataset in datasets:
        dataset_sel = best.df[best.df.dataset == dataset]
        bold_best(dataset_sel)


    print(best.df)
    best.flush()





