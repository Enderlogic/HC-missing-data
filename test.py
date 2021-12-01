from datetime import datetime
import time

import pandas
import random
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

base, bnlearn = importr('base'), importr('bnlearn')
from os import path
import os

pandas2ri.activate()
from lib.accessory import f1, add_missing, find_causes
from lib.hc import hc

result = pandas.DataFrame(columns=['dataset', 'datasize', 'noise', 'method', 'F1', 'SHD', 'cost'])

for i in range(1, 51):
    dag = base.readRDS('sparse_network/' + str(i) + '.rds')
    data = bnlearn.rbn(dag, 10000)
    data = data[random.sample(list(data.columns), data.shape[1])]
    data_missing = add_missing(data, 'MNAR', dag)

    dag_clean = hc(data)
    f1_clean = f1(dag, dag_clean)
    print('F1 for clean data: ', f1_clean)

    for method in ['pw', 'ipw', 'aipw']:
        start = time.time()
        dag_learned = hc(data_missing, method=method)
        cost = time.time() - start
        result.loc[len(result)] = [str(i), 10000, 'MNAR', method, f1(dag, dag_learned),
                                   bnlearn.shd(bnlearn.cpdag(dag), dag_learned)[0], cost]
        print(result.iloc[[-1]].to_string(index=False) + ' time:' + str(datetime.now().strftime("%H:%M:%S")))
a = 1
