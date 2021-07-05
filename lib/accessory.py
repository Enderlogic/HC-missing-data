import itertools
import random
import pingouin as pg

import numpy as np
from graphviz import Digraph
from numba import njit
from rpy2.robjects.packages import importr
from scipy.stats.distributions import chi2
import scipy.stats as st

from lib.score import compute_weights

base, bnlearn = importr('base'), importr('bnlearn')
from rpy2.robjects import pandas2ri

pandas2ri.activate()
random.seed(941214)


def pairwise(data, varnames, vars, cause_list, cache_data, cache_weight, method):
    if method == 'pw':
        W = tuple(sorted([v for v in vars if v in cause_list]))
        if W not in cache_data:
            W_ids = [i for i in range(len(varnames)) if varnames[i] in W]
            if len(W):
                if 'int' in data.dtype.name:
                    cache_data[W] = data[data[:, W_ids].min(axis=1) >= 0]
                else:
                    cache_data[W] = data[~np.isnan(data[:, W_ids]).any(axis=1)]
            else:
                cache_data[W] = data
            cache_weight[W] = np.ones(len(cache_data[W]))
    elif method == 'ipw':
        W = list(vars)
        while True:
            Pa_R_vars = [cause_list[x] for x in W if x in cause_list]
            Pa_R_vars = [x for l in Pa_R_vars for x in l]
            if all(elem in W for elem in Pa_R_vars):
                break
            else:
                W = list(set(W) | set(Pa_R_vars))
        W = tuple(sorted([v for v in W if v in cause_list]))
        if W not in cache_data:
            W_ids = [i for i in range(len(varnames)) if varnames[i] in W]
            if len(W) > 0:
                if 'int' in data.dtype.name:
                    cache_data[W] = data[data[:, W_ids].min(axis=1) >= 0]
                else:
                    cache_data[W] = data[~np.isnan(data[:, W_ids]).any(axis=1)]
                cache_weight[W] = compute_weights(data, varnames, W_ids, cause_list)
            else:
                cache_data[W] = data
                cache_weight[W] = np.ones(len(data))
    elif method == 'aipw':
        Pa_R_vars = [cause_list[x] for x in vars if x in cause_list]
        Pa_R_vars = [x for l in Pa_R_vars for x in l]
        W = tuple(sorted([v for v in vars if v in cause_list.keys()]))
        pa_d = [v for v in Pa_R_vars if v not in W]
        pa_d = [v for v in pa_d if v in cause_list]
        if W not in cache_data:
            if len(W) > 0:
                W_ids = [i for i in range(len(varnames)) if varnames[i] in W]
                if 'int' in data.dtype.name:
                    cache_data[W] = data[data[:, W_ids].min(axis=1) >= 0]
                else:
                    cache_data[W] = data[~np.isnan(data[:, W_ids]).any(axis=1)]

                if pa_d:
                    cache_weight[W] = np.ones(cache_data[W].shape[0])
                else:
                    cache_weight[W] = compute_weights(data, varnames, W_ids, cause_list)
            else:
                cache_data[W] = data
                cache_weight[W] = np.ones(data.shape[0])
    else:
        raise Exception('Unknown variant of HC')
    return cache_data, cache_weight, W


# compute the F1 score of a learned graph given true graph
def f1(dag_true, dag_learned):
    '''
    :param dag_true: true DAG
    :param dag_learned: learned DAG
    :return: the F1 score of learned DAG
    '''
    compare = bnlearn.compare(bnlearn.cpdag(dag_true), bnlearn.cpdag(dag_learned))
    return compare[0][0] * 2 / (compare[0][0] * 2 + compare[1][0] + compare[2][0])


# find the missing mechanism of dataset with missing values
def find_causes(data, test_function='default', alpha=0.01):
    var_miss = data.columns[data.isnull().any()]
    if all(data[var].dtype.name == 'category' for var in data):
        factor = True
        data = data.apply(lambda x: x.cat.codes)
        if test_function == 'default':
            test_function = 'g_test'
    elif all(data[var].dtype.name != 'category' for var in data):
        if test_function == 'default':
            test_function = 'zf_test'
        factor = False
    else:
        raise Exception('Mixed data is not supported.')

    causes = {}
    varnames = data.columns.tolist()
    for var in var_miss:
        causes[var] = list(varnames)
        causes[var].remove(var)
        if factor:
            data['missing'] = data[var] == -1
            data['missing'] = data['missing'].astype('int8')
        else:
            data['missing'] = data[var].isna()
        l = 0
        while len(causes[var]) > l:
            remain_causes = list(causes[var])
            for can in remain_causes:
                cond_set = list(remain_causes)
                cond_set.remove(can)
                for cond in itertools.combinations(cond_set, l):
                    if factor:
                        data_delete = data[(data[[can] + list(cond)] > -1).all(1)].to_numpy()
                    else:
                        data_delete = data.dropna(subset=[can] + list(cond))
                    cols = np.asarray(
                        [data_delete.shape[1] - 1] + [varnames.index(can)] + [varnames.index(x) for x in cond])
                    p_value = globals()[test_function](data_delete, cols)
                    if p_value > alpha:
                        causes[var].remove(can)
                        break
            l += 1
    data.pop('missing')
    return causes


# convert the dag to bnlearn format
def to_bnlearn(dag):
    output = ''
    for var in dag:
        output += '[' + var
        if dag[var]['par']:
            output += '|'
            for par in dag[var]['par']:
                output += par + ':'
            output = output[:-1]
        output += ']'
    return output


def from_bnlearn(dag, varnames):
    output = {}
    for node in varnames:
        output[node] = {}
        output[node]['par'] = list(bnlearn.parents(dag, node))
        output[node]['nei'] = list(
            base.setdiff(base.setdiff(bnlearn.nbr(dag, node), bnlearn.parents(dag, node)), bnlearn.children(dag, node)))
    return output


# statistical G2 test
def g_test(data, cols):
    '''
    :param data: the unique datapoints as a 2-d array, each row is a datapoint, assumed unique
    :param arities: the arities of the variables (=columns) for the contingency table order must match that of `cols`.
    :param cols: the columns (=variables) for the marginal contingency table. columns must be ordered low to high

    :returns : p value
    '''
    arities = np.amax(data, axis=0) + 1
    G, dof = g_counter(data, arities, cols)
    return chi2.sf(G, dof)


@njit(fastmath=True)
def g_counter(data, arities, cols):
    strides = np.empty(len(cols), dtype=np.uint32)
    idx = len(cols) - 1
    stride = 1
    while idx > -1:
        strides[idx] = stride
        stride *= arities[cols[idx]]
        idx -= 1
    N_ijk = np.zeros(stride)
    N_ik = np.zeros(stride)
    N_jk = np.zeros(stride)
    N_k = np.zeros(stride)
    for rowidx in range(data.shape[0]):
        idx_ijk = 0
        idx_ik = 0
        idx_jk = 0
        idx_k = 0
        for i in range(len(cols)):
            idx_ijk += data[rowidx, cols[i]] * strides[i]
            if i != 0:
                idx_jk += data[rowidx, cols[i]] * strides[i]
            if i != 1:
                idx_ik += data[rowidx, cols[i]] * strides[i]
            if (i != 0) & (i != 1):
                idx_k += data[rowidx, cols[i]] * strides[i]
        N_ijk[idx_ijk] += 1
        for j in range(arities[cols[1]]):
            N_ik[idx_ik + j * strides[1]] += 1
        for i in range(arities[cols[0]]):
            N_jk[idx_jk + i * strides[0]] += 1
        for i in range(arities[cols[0]]):
            for j in range(arities[cols[1]]):
                N_k[idx_k + i * strides[0] + j * strides[1]] += 1
    G = 0
    for i in range(stride):
        if N_ijk[i] != 0:
            G += 2 * N_ijk[i] * np.log(N_ijk[i] * N_k[i] / N_ik[i] / N_jk[i])
    dof = (arities[cols[0]] - 1) * (arities[cols[1]] - 1) * strides[1]
    return G, dof


# statistical fisher's z test
def zf_test(data, cols):
    if len(cols) == 2:
        rho = pg.partial_corr(data, data.columns[cols[0]], data.columns[cols[1]])['r'][0]
    elif len(cols) > 2:
        rho = pg.partial_corr(data, data.columns[cols[0]], data.columns[cols[1]], list(data.columns[cols[2:]]))['r'][0]
    else:
        raise Exception('Length of input cols is less than 2')
    z = np.arctanh(rho) * np.sqrt(data.shape[0] - 3 - len(cols) + 2)
    return st.norm.sf(abs(z)) * 2


# statistical pearson test
def pearson_test(data, cols):
    if len(cols) == 2:
        return pg.partial_corr(data, data.columns[cols[0]], data.columns[cols[1]])['p-val'][0]
    elif len(cols) > 2:
        return \
        pg.partial_corr(data, data.columns[cols[0]], data.columns[cols[1]], list(data.columns[cols[2:]]))['p-val'][0]
    else:
        raise Exception('Length of input cols is less than 2')


# plot the bnlearn DAG
def plot(dag, filename):
    dot = Digraph()
    for node in bnlearn.nodes(dag):
        dot.node(node)
        for parent in bnlearn.parents(dag, node):
            if parent not in dot.body:
                dot.node(parent)
            dot.edge(parent, node)
    dot.render(filename, view=True)
