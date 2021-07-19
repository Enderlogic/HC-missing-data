import time
from datetime import datetime

import pandas
import random
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from os import path
import os

pandas2ri.activate()
from lib.accessory import f1, add_missing
from lib.hc import hc

base, bnlearn = importr('base'), importr('bnlearn')
base.set_seed(1009)
random.seed(1009)
datasize_list = [100, 500, 1000, 5000, 10000]
noise_list = ['MCAR', 'MAR', 'MNAR']
method_list = ['pw', 'ipw', 'aipw']
model_list = range(1, 51)

result_path = 'result/test.csv'
# if path.isfile(result_path):
#     result = pandas.read_csv(result_path)
# else:
#     result = pandas.DataFrame(columns=['dataset', 'datasize', 'noise', 'method', 'F1', 'SHD', 'cost'])
result = pandas.DataFrame(columns=['dataset', 'datasize', 'noise', 'method', 'F1', 'SHD', 'cost'])

for model in model_list:
    # load complete dataset
    model_path = 'network/' + str(model) + '.rds'
    dag_true = base.readRDS(model_path)
    cpdag_true = bnlearn.cpdag(dag_true)
    data_path = 'data/' + str(model) + '/' + str(model) + '_Clean.csv'
    if path.isfile(data_path):
        data = pandas.read_csv(data_path, dtype='category')
    else:
        data = bnlearn.rbn(dag_true, max(datasize_list))
        data = data[random.sample(list(data.columns), data.shape[1])]
        os.makedirs('data/' + str(model))
        # data.to_csv(data_path, index = False)
    for datasize in datasize_list:
        if not any((result.dataset == model) & (result.datasize == datasize) & (result.noise == 'Clean') & (
                result.method == 'complete')):
            start = time.time()
            dag_learned = hc(data.head(datasize))
            cost = time.time() - start
            result = result.append({'dataset': model, 'datasize': datasize, 'noise': 'Clean', 'method': 'complete',
                                    'F1': f1(cpdag_true, dag_learned), 'SHD': bnlearn.shd(cpdag_true, dag_learned)[0],
                                    'cost': cost}, ignore_index=True)
            # result.to_csv('result/test.csv', index=False)
            print(result.iloc[[-1]].to_string(index=False) + ' time:' + str(datetime.now().strftime("%H:%M:%S")))
    for noise in noise_list:
        # load missing dataset
        data_path = 'data/' + str(model) + '/' + str(model) + '_' + noise + '.csv'
        if path.isfile(data_path):
            data_missing = pandas.read_csv(data_path, dtype='category')
        else:
            data_missing = add_missing(data, noise, dag_true)
            # data_missing.to_csv(data_path, index = False)
        for datasize in datasize_list:
            for method in method_list:
                if not any((result.dataset == model) & (result.datasize == datasize) & (result.noise == noise) & (
                        result.method == method)):
                    start = time.time()
                    dag_learned = hc(data_missing.head(datasize), method=method)
                    cost = time.time() - start
                    result = result.append({'dataset': model, 'datasize': datasize, 'noise': noise, 'method': method,
                                            'F1': f1(dag_true, dag_learned),
                                            'SHD': bnlearn.shd(cpdag_true, dag_learned)[0], 'cost': cost},
                                           ignore_index=True)
                    # result.to_csv('result/test.csv', index=False)
                    print(result.iloc[[-1]].to_string(index=False) + ' time:' + str(datetime.now().strftime("%H:%M:%S")))