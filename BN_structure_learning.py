<<<<<<< HEAD
import time
from datetime import datetime

import pandas
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
from lib.accessory import f1
from lib.hc import hc

base, bnlearn = importr('base'), importr('bnlearn')
=======
import pandas as pd
import numpy as np
from lib.mmhc import mmhc
from graphviz import Digraph
from lib.evaluation import compare
from lib.accessory import cpdag
import json

# input data for learning
data_set = 'alarm'
data_size = '0.1'
data_training = pd.read_csv('Input/' + data_set + data_size + '.csv')
print('data loaded successfully')

# add noise into original data
# data_noise = data_training.copy()
# epsilon = 0.1
# for var in data_noise:
#     values = data_noise[var].unique()
#     len_values = len(values)
#     noise = epsilon / (len_values - 1) * np.ones((len_values, len_values)) + (1 - epsilon * len_values / (len_values - 1)) * np.identity(len_values)
#     for val in values:
#         data_noise[var][data_noise[var] == val] = np.random.choice(values, len(data_noise[var][data_noise[var] == val]), p = noise[np.where(values == val)[0][0], :])

# learn bayesian network from data
dag = mmhc(data_training, score_function = 'bic', prune = False, threshold = 0.05)

# plot the graph
dot = Digraph()
for k, v in dag.items():
    if k not in dot.body:
        dot.node(k)
    if v:
        for v_ele in v['par']:
            if v_ele not in dot.body:
                dot.node(v_ele)
            dot.edge(v_ele, k)
dot.render('output/' + data_set + data_size + '.gv', view = False)
>>>>>>> 2b4fdb79ca8c83b604ae05bf827b93917f42fdda

datasize_list = [100, 500, 1000, 5000, 10000]
noise_list = ['MCAR', 'MAR', 'MNAR']
method_list = ['pw', 'ipw', 'aipw']
model_list = range(1, 6)
result = pandas.DataFrame(columns=['dataset', 'datasize', 'noise', 'method', 'F1', 'SHD', 'cost'])

<<<<<<< HEAD
for model in model_list:
    # load complete dataset
    model_path = 'networks/' + str(model) + '.rds'
    dag_true = base.readRDS(model_path)
    data = pandas.read_csv('data/' + str(model) + '/' + str(model) + '_Clean.csv', dtype='category')
    # load clean dataset
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
                    print(
                        result.iloc[[-1]].to_string(index=False) + ' time:' + str(datetime.now().strftime("%H:%M:%S")))
result.to_csv('result/results.csv', index=False)
=======
# evaluate the result (comment the following lines if evaluation is not necessary)
with open('Input/' + data_set + '.json') as json_file:
     true_dag = json.load(json_file)

compare_result = compare(cpdag(true_dag), cpdag(dag))
print('Compare the edges with true graph:')
print('\n'.join("{}: {}".format(k, v) for k, v in compare_result.items()))
>>>>>>> 2b4fdb79ca8c83b604ae05bf827b93917f42fdda
