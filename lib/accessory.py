import itertools
import random
import pingouin as pg
from copy import deepcopy

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


# create missing mechanism
# def miss_mechanism(dag, noise='MAR', rom=0.5):
#     '''
#
#     :param dag: true DAG
#     :param noise: type of missing mechanism
#     :param rom: ratio of missing variables
#     :return: parent of missingness for each partially observed variable
#     '''
#     cause_dict = {}
#     if noise == 'MCAR':
#         vars_miss = random.sample(list(bnlearn.nodes(dag)), round(len(bnlearn.nodes(dag)) * rom))
#         for var in vars_miss:
#             cause_dict[var] = []
#     elif noise == 'MAR':
#         varnames = list(bnlearn.nodes(dag))
#         nom = round(len(varnames) * rom)
#         noc = len(varnames) - nom
#         vstructs = np.array(bnlearn.vstructs(dag))
#         vstructs = vstructs.reshape(3, int(len(vstructs) / 3))
#         vars_miss = []
#         vars_comp = []
#         for i in range(vstructs.shape[1]):
#             if len(vars_comp) != noc or (len(vars_comp) == noc and vstructs[1, i] in vars_comp):
#                 if vstructs[0, i] not in vars_miss + vars_comp:
#                     cause_dict[vstructs[0, i]] = [vstructs[1, i]]
#                     vars_miss.append(vstructs[0, i])
#                     vars_comp = list(set(vars_comp + [vstructs[1, i]]))
#                 elif vstructs[2, i] not in vars_miss + vars_comp:
#                     cause_dict[vstructs[2, i]] = [vstructs[1, i]]
#                     vars_miss.append(vstructs[2, i])
#                     vars_comp = list(set(vars_comp + [vstructs[1, i]]))
#                 if len(vars_miss) == nom:
#                     break
#         if len(vars_miss) < nom:
#             vars_miss2 = random.sample([x for x in varnames if x not in vars_miss + vars_comp], nom - len(vars_miss))
#             vars_comp = [x for x in varnames if x not in vars_miss + vars_miss2]
#             for var in vars_miss2:
#                 nbr = list(bnlearn.nbr(dag, var))
#                 if any(item in vars_comp for item in nbr):
#                     cause_dict[var] = random.sample([x for x in nbr if x in vars_comp], 1)
#                 else:
#                     cause_dict[var] = random.sample(vars_comp, 1)
#     elif noise == 'MNAR':
#         varnames = list(bnlearn.nodes(dag))
#         nom = round(len(varnames) * rom)
#         vstructs = np.array(bnlearn.vstructs(dag))
#         vstructs = vstructs.reshape(3, int(len(vstructs) / 3))
#         vars_miss = []
#         for i in range(vstructs.shape[1]):
#             if vstructs[0, i] not in cause_dict:
#                 cause_dict[vstructs[0, i]] = [vstructs[1, i]]
#                 vars_miss = list(set(vars_miss + [vstructs[0, i], vstructs[1, i]]))
#             elif vstructs[2, i] not in cause_dict:
#                 cause_dict[vstructs[2, i]] = [vstructs[1, i]]
#                 vars_miss = list(set(vars_miss + [vstructs[2, i], vstructs[1, i]]))
#             if len(vars_miss) >= nom:
#                 break
#         if len(vars_miss) < nom:
#             vars_miss2 = random.sample([x for x in varnames if x not in vars_miss], nom - len(vars_miss))
#             vars_comp = [x for x in varnames if x not in vars_miss + vars_miss2]
#             for var in vars_miss2:
#                 nbr = list(bnlearn.nbr(dag, var))
#                 if any(item not in vars_comp for item in nbr):
#                     cause_dict[var] = random.sample([x for x in nbr if x not in vars_comp + [var]], 1)
#                 else:
#                     cause_dict[var] = random.sample([x for x in varnames if x not in vars_comp + [var]], 1)
#     else:
#         raise Exception('noise ' + noise + ' is undefined.')
#     return cause_dict

# create missing mechanism
def miss_mechanism(dag, noise='MAR', rom=0.5):
    '''

    :param dag: true DAG
    :param noise: type of missing mechanism
    :param rom: ratio of missing variables
    :return: parent of missingness for each partially observed variable
    '''
    cause_dict = {}
    vars_miss = random.sample(list(bnlearn.nodes(dag)), round(len(bnlearn.nodes(dag)) * rom))
    vars_comp = [v for v in list(bnlearn.nodes(dag)) if v not in vars_miss]
    if noise == 'MCAR':
        for var in vars_miss:
            cause_dict[var] = []
    elif noise == 'MAR':
        for var in vars_miss:
            neighbour = list(bnlearn.nbr(dag, var))
            neighbour = [v for v in list(neighbour) if v in vars_comp]
            if len(neighbour) > 0:
                cause_dict[var] = random.sample(neighbour, 1)
            else:
                cause_dict[var] = random.sample(vars_comp, 1)
    elif noise == 'MNAR':
        for var in vars_miss:
            neighbour = list(bnlearn.nbr(dag, var))
            if random.uniform(0, 1) > 0.5:
                neighbour = [v for v in list(neighbour) if v in vars_comp]
                if len(neighbour) > 0:
                    cause_dict[var] = random.sample(neighbour, 1)
                else:
                    cause_dict[var] = random.sample(vars_comp, 1)
            else:
                neighbour = [v for v in list(neighbour) if v in vars_miss]
                if len(neighbour) > 0:
                    cause_dict[var] = random.sample(neighbour, 1)
                else:
                    cause_dict[var] = random.sample([v for v in list(vars_miss) if v is not var], 1)
    else:
        raise Exception('noise ' + noise + ' is undefined.')
    return cause_dict


# add missing value in dataset
def add_missing(data, noise, dag, m_min=0.1, m_max=0.6):
    data_missing = deepcopy(data)
    cause_dict = miss_mechanism(dag, noise)
    for var in cause_dict.keys():
        if len(cause_dict[var]) == 0:
            m = np.random.uniform(m_min, m_max)
            data_missing[var][np.random.uniform(size=len(data)) < m] = np.nan
        else:
            for cause in cause_dict[var]:
                if data[cause].dtype == 'category':
                    state_m = data[cause].mode()[0]
                    data_missing[var][(np.random.uniform(size=len(data)) < m_max) & (data[cause] == state_m)] = np.nan
                    data_missing[var][(np.random.uniform(size=len(data)) < m_min) & (data[cause] != state_m)] = np.nan
                elif data[cause].dtype == 'float' or data[cause].dtype == 'int':
                    thres = data[cause].quantile(0.2)
                    data_missing[var][(np.random.uniform(size=len(data)) < m_max) & (data[cause] < thres)] = np.nan
                    data_missing[var][(np.random.uniform(size=len(data)) < m_min) & (data[cause] >= thres)] = np.nan
                else:
                    raise Exception('data type ' + data[cause].dtype + ' is not supported.')
    return data_missing


# pairwise delete input data based on vars and method
def pairwise(data, varnames, vars, cause_list, cache_data, cache_weight, method):
    '''
    :param data: input data (int type for categorical data and float type for continuous data)
    :param varnames: the name of variables (list type)
    :param vars: the data cases with any missing value in vars will be removed from input data
    :param cause_list: detected parents of missingness of partially observed variables
    :param cache_data: cached pairwise deleted data sets that constructed in previous iteration
    :param cache_weight: cached weights that constructed in previous iteration
    :param method: method to deal with missing values, pw: purely pairwise deletion; ipw: inverse probability weighting; aipw: adaptive inverse probability weighting
    :return: cached pairwise deleted datasets, cached weights, and partially observed variables in vars (set type)
    '''
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
    '''

    :param data: input data (int type for categorical data and float type for continuous data)
    :param test_function: statistical test used to determine conditional independence between variables. Current support g_test (for discrete data), pearson_test and zf_test (for continuous data)
    :param alpha: significance level of test
    :return: a list of detected parents of missingenss for every partially observed variable
    '''
    var_miss = data.columns[data.isnull().any()]
    if all(data[var].dtype.name == 'category' for var in data):
        factor = True
        data = data.apply(lambda x: x.cat.codes)
        if test_function == 'default':
            test_function = 'g_test'
    elif all(data[var].dtype.name != 'category' for var in data):
        if test_function == 'default':
            test_function = 'pearson_test'
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


# convert bnlearn format to my format
def from_bnlearn(dag, varnames):
    output = {}
    for node in varnames:
        output[node] = {}
        output[node]['par'] = list(bnlearn.parents(dag, node))
        output[node]['nei'] = list(
            base.setdiff(base.setdiff(bnlearn.nbr(dag, node), bnlearn.parents(dag, node)), bnlearn.children(dag, node)))
    return output


# random orient a CPDAG to a DAG
def random_orient(cpdag):
    undirected_edges = []
    for var in cpdag:
        for nei in cpdag[var]['nei']:
            edge = sorted([var, nei])
            if edge not in undirected_edges:
                undirected_edges.append(edge)
    random.shuffle(undirected_edges)
    orient_state = []
    orient_history = []
    dag = deepcopy(cpdag)
    index = 0
    while len(undirected_edges):
        edge = undirected_edges[index]
        sin_flag_temp, sin_direction_temp = sin_path_check(dag, edge[0], edge[1])
        v_flag_temp, v_direction_temp = v_check(dag, edge[0], edge[1])
        dag[edge[0]]['nei'].remove(edge[1])
        dag[edge[1]]['nei'].remove(edge[0])
        if (not sin_flag_temp) & (not v_flag_temp):
            orient_history.append(random.randint(0, 1))
            if orient_history[-1] == 0:
                dag[edge[0]]['par'].append(edge[1])
            else:
                dag[edge[1]]['par'].append(edge[0])
            orient_state.append(0)
            index += 1
        elif sin_flag_temp & (not v_flag_temp):
            dag[sin_direction_temp[1]]['par'].append(sin_direction_temp[0])
            if sin_direction_temp[1] == edge[0]:
                orient_history.append(0)
            else:
                orient_history.append(1)
            orient_state.append(1)
            index += 1
        elif (not sin_flag_temp) & v_flag_temp & (v_direction_temp != 'both'):
            dag[v_direction_temp[1]]['par'].append(v_direction_temp[0])
            if v_direction_temp[1] == edge[0]:
                orient_history.append(0)
            else:
                orient_history.append(1)
            orient_state.append(1)
            index += 1
        elif sin_flag_temp & v_flag_temp & (v_direction_temp == sin_direction_temp):
            dag[v_direction_temp[1]]['par'].append(v_direction_temp[0])
            if v_direction_temp[1] == edge[0]:
                orient_history.append(0)
            else:
                orient_history.append(1)
            orient_state.append(1)
            index += 1
        else:
            if 0 in orient_state[::-1]:
                last = len(orient_state) - 1 - orient_state[::-1].index(0)
                dag = deepcopy(cpdag)
                orient_history_temp = []
                for i in range(last):
                    edge = undirected_edges[i]
                    dag[edge[0]]['nei'].remove(edge[1])
                    dag[edge[1]]['nei'].remove(edge[0])
                    if orient_history[i] == 0:
                        dag[edge[0]]['par'].append(edge[1])
                    else:
                        dag[edge[1]]['par'].append(edge[0])
                    orient_history_temp.append(orient_history[i])
                edge = undirected_edges[last]
                dag[edge[0]]['nei'].remove(edge[1])
                dag[edge[1]]['nei'].remove(edge[0])
                if orient_history[last] == 0:
                    dag[edge[1]]['par'].append(edge[0])
                    orient_history_temp.append(1)
                else:
                    dag[edge[0]]['par'].append(edge[1])
                    orient_history_temp.append(0)
                index = last + 1
                orient_state = orient_state[: last + 1]
                orient_state[last] = 1
                orient_history = deepcopy(orient_history_temp)
            else:
                orient_history.append(random.randint(0, 1))
                if orient_history[-1] == 0:
                    dag[edge[0]]['par'].append(edge[1])
                else:
                    dag[edge[1]]['par'].append(edge[0])
                orient_state.append(0)
                index += 1

        if index == len(undirected_edges):
            break
    return dag


# single direction path check
def sin_path_check(dag, var1, var2):
    sin_flag = False
    sin_direction = None
    # check single direction path var1 -> ... -> var2
    unchecked = deepcopy(dag[var2]['par'])
    checked = []
    while unchecked:
        if sin_flag:
            break
        unchecked_copy = deepcopy(unchecked)
        for dag_par in unchecked_copy:
            if var1 in dag[dag_par]['par']:
                sin_flag = True
                sin_direction = [var1, var2]
                break
            else:
                for key in dag[dag_par]['par']:
                    if key not in checked:
                        unchecked.append(key)
            unchecked.remove(dag_par)
            checked.append(dag_par)

    # check single direction path var2 -> ... -> var1
    if not sin_flag:
        unchecked = deepcopy(dag[var1]['par'])
        checked = []
        while unchecked:
            if sin_flag:
                break
            unchecked_copy = deepcopy(unchecked)
            for dag_par in unchecked_copy:
                if var2 in dag[dag_par]['par']:
                    sin_flag = True
                    sin_direction = [var2, var1]
                    break
                else:
                    for key in dag[dag_par]['par']:
                        if key not in checked:
                            unchecked.append(key)
                unchecked.remove(dag_par)
                checked.append(dag_par)
    return sin_flag, sin_direction


# v-structure check
def v_check(dag, var1, var2):
    v_flag1 = False
    v_flag2 = False
    if len(dag[var1]['par']):
        for par in dag[var1]['par']:
            if (var2 not in dag[par]['nei']) and (var2 not in dag[par]['par']) and (par not in dag[var2]['par']):
                v_flag1 = True
                break
    if len(dag[var2]['par']):
        for par in dag[var2]['par']:
            if (var1 not in dag[par]['nei']) & (var1 not in dag[par]['par']) & (par not in dag[var1]['par']):
                v_flag2 = True
                break
    if v_flag1 & (not v_flag2):
        return v_flag1, [var1, var2]
    elif (not v_flag1) & v_flag2:
        return v_flag2, [var2, var1]
    elif v_flag1 & v_flag2:
        return True, 'both'
    else:
        return False, None


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
            pg.partial_corr(data, data.columns[cols[0]], data.columns[cols[1]], list(data.columns[cols[2:]]))['p-val'][
                0]
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
