#This file is based in the code for the computation of partitions
#for the datasets that were used in the experiment

# NOTE: Originally, this code was run in the Yandex DataSphere Virtual environment
# this code requires next package that has the original code for the algorithms:
#%pip install --upgrade --force-reinstall git+https://github.com/glendawur/MirCl

import time
import gzip
import pickle
import multiprocessing
from glob import glob

import os
import numpy as np
from tqdm import tqdm

from mircl.miscellaneous import centering
from mircl import experiment, clustering



def process(dataset, path_in, path_out, idx, is_kmeans = True):
    """
    This function is used for the processing of one dataset

    dataset - file with dataset name
    path_in 
    """

    #Note: all the datasets were edited so the last column is the original labels
    data = np.genfromtxt(f'{path_in}/{dataset}', delimiter=',', skip_header = 1)
    #Dataset was centered and normalized between -0.5 and 0.5 
    X = centering(data[:,:-1], normalize = True)
    Y = data[:,-1]
    
    if is_kmeans:
        km_pipe = experiment.AlgorithmPipeline(data = X,
                                               algorithm=clustering.Kmeans)

        L_km, _ = km_pipe.run(k_range = np.arange(2,31),
                                         exec_number=50,
                                         max_iter = 60,
                                         iter_count = False)

        np.save(f'{path_out}/{dataset[:-4]}_{idx}_kmeans_L', L_km)
    
    else:
        rs_pipe = experiment.AlgorithmPipeline(data = X,
                                               algorithm=clustering.RandomSwap)
        L_rs, _ = rs_pipe.run(k_range = np.arange(2,31),
                                         exec_number=50,
                                         max_iter = 30,
                                         iter_count = False)

        np.save(f'{path_out}/{dataset[:-4]}_{idx}_rswap_L', L_rs)

 
p = multiprocessing.Pool()

for folder, iterations in [('uci_data', 50), ('synth_data', 1), ('benchmark_data', 50)]:
    for f in glob(os.path.join(folder, "*.csv")):
        for i in range(iterations):
            for method in [True, False]:
                # launch a process for each file (ish).
                # The result will be approximately one process per CPU core available.
                p.apply_async(process, [f[len(folder)+1:], folder, 'partitions', i, method]) 

p.close()
p.join()