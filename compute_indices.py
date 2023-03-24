#This file is based in the code for the computation of indices
#for the datasets that were used in the experiment
#to run this code, partitions should be computed beforehand

# NOTE: Originally, this code was run in the Yandex DataSphere Virtual environment
# this code requires next package that has the original code for the algorithms:
#%pip install --upgrade --force-reinstall git+https://github.com/glendawur/MirCl

import time
import gzip
import pickle
import multiprocessing
from glob import glob
import re
import os

import numpy as np
from tqdm import tqdm

from mircl.metrics import ami, ari, find_optimal, wss_matrix, bss_matrix
from mircl.metrics import calinski_harabasz_matrix, elbow, hartigan, wb_index_matrix
from mircl.metrics import xu_index_matrix, silhouette_matrix, silhouette_wss

def process_partition(dataset, idx, path_in_dataset, algorithm, path_out):
    #upload data block
    data = np.genfromtxt(f'{path_in_dataset}/{dataset}', delimiter=',', skip_header=1)
    X = centering(data[:,:-1], normalize = True)
    Y = data[:,-1]
    

    res = dict()
    if path_in_dataset == 'synth_data':
        res['dataset'] = 'synth'
        res['id'] = int(dataset[:-4].split('_')[-1][1:])
        res['M'] = int(dataset[:-4].split('_')[1][1:])
        res['N'] = 2500
        res['k'] = int(dataset[:-4].split('_')[3][1:])
        res['a'] = float(dataset[:-4].split('_')[2][1:])
        L = np.load(f'partitions/{dataset[:-4]}_{algorithm}_L.npy')
        res['L'] = L.shape
    elif path_in_dataset == 'uci_data':
        res['dataset'] = re.sub(r'[^a-zA-Z ]+', '', dataset[:-4])
        res['id'] = idx
        res['M'] = X.shape[1]
        res['N'] = X.shape[0]
        res['k'] = int(np.unique(Y).shape[0])
        res['a'] = '-'
        L = np.load(f'partitions/{dataset[:-4]}_{idx}_{algorithm}_L.npy')
        res['L'] = L.shape
    else:
        res['dataset'] = re.sub(r'[^a-zA-Z ]+', '', dataset[:-4])
        res['id'] = idx
        res['M'] = X.shape[1]
        res['N'] = X.shape[0]
        res['k'] = int(np.unique(Y).shape[0])
        if res['dataset'] = 'g':
            res['dataset'] = 'g2'
            res['var'] = int(dataset[:-4].split('-')[2])
        elif res['dataset'] = 's'
            res['var'] = int(dataset[1])
        else:
            res['var'] = '-'
        L = np.load(f'partitions/{dataset[:-4]}_{idx}_{algorithm}_L.npy')
        res['L'] = L.shape
    
    
    for dist in ['conventional', 'euclidean']:
        t1 = time.process_time()
        SSW = wss_matrix(centering(X), L, dist)
        SSB = bss_matrix(centering(X), L, dist)
        t2 = time.process_time()
        sswb_t = t2 - t1
        for aggregation in ['mean', 'optimum']:
            for index in ['elbow', 'calinski_harabasz', 'hartigan', 'wb_index', 'xu_index']:
                if aggregation == 'mean':
                    agg_func = np.mean
                    agg_name = agg_func.__name__
                else:
                    if index == 'calinski_harabasz':
                        agg_func = np.max
                        agg_name = agg_func.__name__
                    else:
                        agg_func = np.min
                        agg_name = agg_func.__name__

                if index == 'elbow':
                    for levels in [(1,1), (2,2), (1,2), (2,1), (3,3)]:
                        t1 = time.process_time()
                        k_p, i_p = find_optimal(elbow(SSW=SSW, levels=levels, aggregation=agg_func), 
                                                index, np.arange(2,31))
                        t2 = time.process_time()
                        output = dict()
                        out_t = t2 - t1
                        output['K'] = k_p
                        output['Time'] = out_t+sswb_t
                        output['ARI'] = ari(Y, L[i_p,SSW[i_p].argmin()], 'ari')
                        output['AMI'] = ami(Y, L[i_p,SSW[i_p].argmin()], 'nmi')                
                        res[(index, levels, dist, agg_name)] = output
                else:
                    if index == 'hartigan':
                        t1 = time.process_time()
                        hk = find_optimal(hartigan(centering(X), L, SSW, agg_func), index, np.arange(2,31))    
                        t2 = time.process_time()
                        if type(hk) == tuple:
                            k, i = hk
                        else:
                            k = hk
                    elif index == 'calinski_harabasz':
                        t1 = time.process_time()
                        k, i = find_optimal(calinski_harabasz_matrix(centering(X), L, SSW, SSB, agg_func),
                                            index, np.arange(2,31))
                        t2 = time.process_time()
                    elif index == 'wb_index':
                        t1 = time.process_time()
                        k, i = find_optimal(wb_index_matrix(L, SSW, SSB, agg_func),
                                            index, np.arange(2,31))
                        t2 = time.process_time()
                    else:
                        t1 = time.process_time()
                        k, i = find_optimal(xu_index_matrix(centering(X), L, SSW, agg_func),
                                            index, np.arange(2,31))
                        t2 = time.process_time()
                    output = {}
                    out_t = t2 - t1
                    output['K'] = k
                    output['Time'] = out_t+sswb_t
                    if ~np.isnan(k):
                        output['ARI'] = ari(Y, L[i,SSW[i].argmin()], 'ari')
                        output['AMI'] = ami(Y, L[i,SSW[i].argmin()], 'nmi')                
                    else:
                        output['ARI'] = np.nan
                        output['AMI'] = np.nan
                    res[(index, dist, agg_name)] = output
    
    t1 = time.process_time()
    silh = silhouette_wss(centering(X), L, SSW, 'mean', np.argmin)
    k, i = find_optimal(silh, 'silhouette', np.arange(2,31))
    t2 = time.process_time()
    output = {}
    out_t = t2 - t1
    output['K'] = k
    output['Time'] = out_t+sswb_t
    output['ARI'] = ari(Y, L[i,SSW[i].argmin()], 'ari')
    output['AMI'] = ami(Y, L[i,SSW[i].argmin()], 'nmi')   
    res[('silhouette', 'wss_min')] = output

    if path_in_dataset = 'synth_data':    
        with gzip.open(f'{path_out}/{dataset[:-4]}_{algorithm}.pkl.gz', 'wb') as f:
            pickle.dump(res, f)
    else:
        with gzip.open(f'{path_out}/{dataset[:-4]}_{idx}_{algorithm}.pkl.gz', 'wb') as f:
            pickle.dump(res, f)


p = multiprocessing.Pool()
for folder, iterations in [('uci_data', 50), ('synth_data', 1), ('benchmark_data', 50)]:
    for dataset in glob(os.path.join(folder, "*.csv")): 
        for idx in range(iterations):
            for algorithm in ['kmeans', 'rswap']:
                p.apply_async(process_partition, [dataset[len(folder)+1:], idx, folder, algorithm, 'analysis'])
p.close()
p.join()