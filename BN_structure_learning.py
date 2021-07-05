import time
from datetime import datetime

import numpy as np
import pandas
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from os import path
pandas2ri.activate()
from lib.accessory import f1
from lib.hc import hc

base, bnlearn = importr('base'), importr('bnlearn')

np.random.seed(941214)

datasize_list = [100, 500, 1000, 5000, 10000]
noise_list = ['MCAR', 'MAR', 'MNAR']
method_list = ['pw', 'ipw', 'aipw']
model_list = range(1, 5)

if path.isfile('result/result.csv'):
    result = pandas.read_csv('result/result.csv')
else:
    result = pandas.DataFrame(columns=['dataset', 'datasize', 'noise', 'method', 'F1', 'SHD', 'cost'])

for model in model_list:
    # load complete dataset
    model_path = 'network/' + str(model) + '.rds'
    dag_true = base.readRDS(model_path)
    data = pandas.read_csv('data/' + str(model) + '/' + str(model) + '_Clean.csv', dtype='category')
    for datasize in datasize_list:
        if not any(
                (result['dataset'] == model) & (result['datasize'] == datasize) & (result['noise'] == 'Clean') & (
                        result['method'] == 'complete')):
            start = time.time()
            dag_learned = hc(data.head(datasize))
            cost = time.time() - start
            result = result.append({'dataset': model, 'datasize': datasize, 'noise': 'Clean', 'method': 'complete',
                                    'F1': f1(dag_true, dag_learned),
                                    'SHD': bnlearn.shd(bnlearn.cpdag(dag_true), dag_learned)[0], 'cost': cost},
                                   ignore_index=True)
            result.to_csv('result/result.csv', index=False)
            print(
                result.iloc[[-1]].to_string(index=False) + ' time:' + str(datetime.now().strftime("%H:%M:%S")))
    for noise in noise_list:
        # load missing dataset
        data_missing = pandas.read_csv('data/' + str(model) + '/' + str(model) + '_' + noise + '.csv', dtype='category')
        for datasize in datasize_list:
            for method in method_list:
                if not any(
                        (result['dataset'] == model) & (result['datasize'] == datasize) & (result['noise'] == noise) & (
                                result['method'] == method)):
                    start = time.time()
                    dag_learned = hc(data_missing.head(datasize), method=method)
                    cost = time.time() - start
                    result = result.append({'dataset': model, 'datasize': datasize, 'noise': noise, 'method': method,
                                            'F1': f1(dag_true, dag_learned),
                                            'SHD': bnlearn.shd(bnlearn.cpdag(dag_true), dag_learned)[0], 'cost': cost},
                                           ignore_index=True)
                    result.to_csv('result/result.csv', index=False)
                    print(
                        result.iloc[[-1]].to_string(index=False) + ' time:' + str(datetime.now().strftime("%H:%M:%S")))