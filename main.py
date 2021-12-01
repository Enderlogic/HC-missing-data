import pandas
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

base, bnlearn = importr('base'), importr('bnlearn')

from lib.hc import hc
pandas2ri.activate()

dag = base.readRDS('network/alarm.rds')
data = pandas.read_csv('data/alarm.csv')

dag_learned = hc(data)

print('SHD score of the learned DAG is: ' + bnlearn.shd(bnlearn.cpdag(dag), dag_learned))