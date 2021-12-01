import time

import numpy as np
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import ListVector

pandas2ri.activate()
from lib.accessory import to_bnlearn, find_causes, pairwise
from lib.score import local_score

base, bnlearn = importr('base'), importr('bnlearn')
from copy import deepcopy


def check_cycle(vi, vj, dag):
    # whether adding or orientating edge vi->vj would cause cycle. In other words, this function check whether there is a direct path from vj to vi except the possible edge vi<-vj
    underchecked = [x for x in dag[vi]['par'] if x != vj]
    checked = []
    cyc_flag = False
    while underchecked:
        if cyc_flag:
            break
        underchecked_copy = list(underchecked)
        for vk in underchecked_copy:
            if dag[vk]['par']:
                if vj in dag[vk]['par']:
                    cyc_flag = True
                    break
                else:
                    for key in dag[vk]['par']:
                        if key not in checked + underchecked:
                            underchecked.append(key)
            underchecked.remove(vk)
            checked.append(vk)
    return cyc_flag


def hc(data, method='complete', score_function='default', debug=False):
    '''
    :param data: the training data used for learn BN
    :param method: the method for dealing with missing values, including:
                   lw (listwise deletion)
                   pw (pairwise deletion)
                   ipw (inverse probability weighting)
                   aipw (adaptive ipw)
                   complete (regular)
    :param score_function: score function, including:
                   bic (Bayesian Information Criterion for discrete variable)
                   nal (Node Average Likelihood for discrete variable)
    :return: the learned BN
    '''
    if score_function == 'default':
        if all(data[var].dtype.name == 'category' for var in data):
            score_function = 'bic'
        else:
            score_function = 'bic_g'
    dc = data.nunique() == 1
    dc = dc[dc].index.values
    data = data.drop(dc, axis = 1)
    if method == 'lw':
        data_listwise = data.dropna()
        dag = bnlearn.hc(data_listwise, score_function=score_function, debug=debug)
    elif method == 'pw' or method == 'ipw' or method == 'aipw':
        # calculate the run time
        time_total = time.time()
        time_checkcycle = 0
        time_score = 0
        time_delete = 0
        # initialize the candidate set for each variable
        candidate = {}
        dag = {}
        # store the computed scores
        cache_score = {}
        cache_data = {}
        cache_weight = {}
        start_time = time.time()
        varnames = data.columns.tolist()
        if method == 'pw':
            cause_list = data.columns[data.isnull().any()].tolist()
        else:
            cause_list = find_causes(data)
        if all(data[var].dtype.name == 'category' for var in data):
            data = data.apply(lambda x: x.cat.codes).to_numpy()
        elif all(data[var].dtype.name != 'category' for var in data):
            data = data.to_numpy()
        else:
            raise Exception('Mixed data is not supported.')
        for var in varnames:
            candidate[var] = list(varnames)
            candidate[var].remove(var)
            dag[var] = {}
            dag[var]['par'] = []
            dag[var]['nei'] = []
            cache_score[var] = {}
        time_preprocess = time.time() - start_time

        diff = 1
        cache_dag = [deepcopy(dag)]
        while diff > 0:
            diff = 0
            edge_candidate = []
            for vi in varnames:
                # attempt to add edges vi->vj
                for vj in candidate[vi]:
                    # calculate the time for checking cycles
                    start_time = time.time()
                    cyc_flag = check_cycle(vi, vj, dag)
                    time_checkcycle += time.time() - start_time
                    dag[vj]['par'] = sorted(dag[vj]['par'] + [vi])
                    if not cyc_flag and dag not in cache_dag:
                        dag[vj]['par'].remove(vi)
                        # perform pairwise deletion based on variables with different parents
                        start_time = time.time()
                        vars = [vi] + dag[vj]['par'] + [vj]
                        cache_data, cache_weight, W = pairwise(data, varnames, vars, cause_list, cache_data,
                                                               cache_weight, method)
                        time_delete += time.time() - start_time
                        # calculate the time for computing the score
                        start_time = time.time()
                        # compute the local score for the current graph
                        par_cur = tuple(dag[vj]['par'])
                        if par_cur not in cache_score[vj]:
                            cache_score[vj][par_cur] = {}
                        if W not in cache_score[vj][par_cur]:
                            cols = [varnames.index(x) for x in (vj, ) + par_cur]
                            cache_score[vj][par_cur][W] = local_score(cache_data[W], cols, score_function,
                                                                      cache_weight[W])
                        # compute the local score for the searching graph
                        par_sea = tuple(sorted(dag[vj]['par'] + [vi]))
                        if par_sea not in cache_score[vj]:
                            cache_score[vj][par_sea] = {}
                        if W not in cache_score[vj][par_sea]:
                            cols = [varnames.index(x) for x in (vj, ) + par_sea]
                            cache_score[vj][par_sea][W] = local_score(cache_data[W], cols, score_function,
                                                                      cache_weight[W])
                        if cache_score[vj][par_sea][W] != np.nan:
                            diff_temp = cache_score[vj][par_sea][W] - cache_score[vj][par_cur][W]
                            if debug:
                                print(vi, vj, diff_temp, 'a')
                            if diff_temp - diff > 1e-10:
                                diff = diff_temp
                                edge_candidate = [vi, vj, 'a']
                        time_score += time.time() - start_time
                    else:
                        dag[vj]['par'].remove(vi)
                parents = list(dag[vi]['par'])
                for par_vi in parents:
                    # attempt to reverse edges from vi<-par_vi to vi->par_vi
                    # calculate the time for checking cycles
                    start_time = time.time()
                    cyc_flag = check_cycle(vi, par_vi, dag)
                    time_checkcycle += time.time() - start_time
                    dag[vi]['par'].remove(par_vi)
                    dag[par_vi]['par'] = sorted(dag[par_vi]['par'] + [vi])
                    if not cyc_flag and dag not in cache_dag:
                        dag[vi]['par'] = sorted(dag[vi]['par'] + [par_vi])
                        dag[par_vi]['par'].remove(vi)
                        # do pairwise deletion on dataset based on variables with different parents
                        start_time = time.time()
                        vars = list(set([vi] + dag[vi]['par'] + [par_vi] + dag[par_vi]['par']))
                        cache_data, cache_weight, W = pairwise(data, varnames, vars, cause_list, cache_data,
                                                               cache_weight, method)
                        time_delete += time.time() - start_time
                        # calculate the time for computing the score
                        start_time = time.time()
                        # compute the local score for the current graph
                        par_cur_par_vi = tuple(dag[par_vi]['par'])
                        if par_cur_par_vi not in cache_score[par_vi]:
                            cache_score[par_vi][par_cur_par_vi] = {}
                        if W not in cache_score[par_vi][par_cur_par_vi]:
                            cols = [varnames.index(x) for x in (par_vi, ) + par_cur_par_vi]
                            cache_score[par_vi][par_cur_par_vi][W] = local_score(cache_data[W], cols, score_function,
                                                                                 cache_weight[W])
                        par_cur_vi = tuple(dag[vi]['par'])
                        if par_cur_vi not in cache_score[vi]:
                            cache_score[vi][par_cur_vi] = {}
                        if W not in cache_score[vi][par_cur_vi]:
                            cols = [varnames.index(x) for x in (vi, ) + par_cur_vi]
                            cache_score[vi][par_cur_vi][W] = local_score(cache_data[W], cols, score_function,
                                                                         cache_weight[W])
                        # compute the local score for the searching graph
                        par_sea_par_vi = tuple(sorted(dag[par_vi]['par'] + [vi]))
                        if par_sea_par_vi not in cache_score[par_vi]:
                            cache_score[par_vi][par_sea_par_vi] = {}
                        if W not in cache_score[par_vi][par_sea_par_vi]:
                            cols = [varnames.index(x) for x in (par_vi, ) + par_sea_par_vi]
                            cache_score[par_vi][par_sea_par_vi][W] = local_score(cache_data[W], cols,
                                                                                 score_function, cache_weight[W])
                        par_sea_vi = tuple([x for x in dag[vi]['par'] if x != par_vi])
                        if par_sea_vi not in cache_score[vi]:
                            cache_score[vi][par_sea_vi] = {}
                        if W not in cache_score[vi][par_sea_vi]:
                            cols = [varnames.index(x) for x in (vi, ) + par_sea_vi]
                            cache_score[vi][par_sea_vi][W] = local_score(cache_data[W], cols, score_function,
                                                                         cache_weight[W])
                        if cache_score[vi][par_cur_vi][W] != np.nan and cache_score[par_vi][par_sea_par_vi][
                            W] != np.nan:
                            diff_temp = cache_score[vi][par_sea_vi][W] + cache_score[par_vi][par_sea_par_vi][W] - \
                                        cache_score[vi][par_cur_vi][W] - cache_score[par_vi][par_cur_par_vi][W]
                            if diff_temp - diff > 1e-10:
                                diff = diff_temp
                                edge_candidate = [vi, par_vi, 'r']
                            if debug:
                                print(vi, par_vi, diff_temp, 'r')
                        time_score += time.time() - start_time
                    else:
                        dag[vi]['par'] = sorted(dag[vi]['par'] + [par_vi])
                        dag[par_vi]['par'].remove(vi)
                    # attempt to delete edges vi<-par_vi
                    dag[vi]['par'].remove(par_vi)
                    if dag not in cache_dag:
                        dag[vi]['par'] = sorted(dag[vi]['par'] + [par_vi])
                        # do pairwise deletion on dataset based on variables with different parents
                        start_time = time.time()
                        vars = [vi] + dag[vi]['par']
                        cache_data, cache_weight, W = pairwise(data, varnames, vars, cause_list, cache_data,
                                                               cache_weight, method)
                        time_delete += time.time() - start_time
                        # calculate the time for computing the score
                        start_time = time.time()
                        # compute the local score for the current graph
                        par_cur = tuple(dag[vi]['par'])
                        if par_cur not in cache_score[vi]:
                            cache_score[vi][par_cur] = {}
                        if W not in cache_score[vi][par_cur]:
                            cols = [varnames.index(x) for x in (vi, ) + par_cur]
                            cache_score[vi][par_cur][W] = local_score(cache_data[W], cols, score_function,
                                                                      cache_weight[W])
                        # compute the local score for the searching graph
                        par_sea = tuple([x for x in dag[vi]['par'] if x != par_vi])
                        if par_sea not in cache_score[vi]:
                            cache_score[vi][par_sea] = {}
                        if W not in cache_score[vi][par_sea]:
                            cols = [varnames.index(x) for x in (vi, ) + par_sea]
                            cache_score[vi][par_sea][W] = local_score(cache_data[W], cols, score_function,
                                                                      cache_weight[W])
                        if cache_score[vi][par_cur][W] != np.nan:
                            diff_temp = cache_score[vi][par_sea][W] - cache_score[vi][par_cur][W]
                            if diff_temp - diff > 1e-10:
                                diff = diff_temp
                                edge_candidate = [par_vi, vi, 'd']
                            if debug:
                                print(par_vi, vi, diff_temp, 'd')
                        time_score += time.time() - start_time
                    else:
                        dag[vi]['par'] = sorted(dag[vi]['par'] + [par_vi])
            if edge_candidate:
                if edge_candidate[-1] == 'a':
                    dag[edge_candidate[1]]['par'] = sorted(dag[edge_candidate[1]]['par'] + [edge_candidate[0]])
                    candidate[edge_candidate[0]].remove(edge_candidate[1])
                    candidate[edge_candidate[1]].remove(edge_candidate[0])
                elif edge_candidate[-1] == 'r':
                    dag[edge_candidate[1]]['par'] = sorted(dag[edge_candidate[1]]['par'] + [edge_candidate[0]])
                    dag[edge_candidate[0]]['par'].remove(edge_candidate[1])
                elif edge_candidate[-1] == 'd':
                    dag[edge_candidate[1]]['par'].remove(edge_candidate[0])
                    candidate[edge_candidate[0]].append(edge_candidate[1])
                    candidate[edge_candidate[1]].append(edge_candidate[0])
                if debug:
                    print('best operation is:', edge_candidate, diff)
                cache_dag.append(deepcopy(dag))
        dag = bnlearn.model2network(to_bnlearn(dag))
        time_total = time.time() - time_total
        if debug:
            print('total cost: %.2f seconds' % time_total)
            print('preprocess cost: %.2f%%' % (time_preprocess / time_total * 100))
            print('check cycle cost: %.2f%%' % (time_checkcycle / time_total * 100))
            print('compute score cost: %.2f%%' % (time_score / time_total * 100))
            print('pairwise deletion cost: %.2f%%' % (time_delete / time_total * 100))
            print('others cost: %.2f%%' % (
                    (time_total - time_preprocess - time_checkcycle - time_score - time_delete) / time_total * 100))
    elif method == 'complete':
        # calculate the run time
        time_total = time.time()
        time_checkcycle = 0
        time_score = 0
        # initialize the candidate parents-set for each variable
        candidate = {}
        dag = {}
        cache = {}
        start_time = time.time()
        varnames = data.columns.tolist()
        if all(data[var].dtype.name == 'category' for var in data):
            data = data.apply(lambda x: x.cat.codes).to_numpy()
        elif all(data[var].dtype.name != 'category' for var in data):
            data = data.to_numpy()
        else:
            raise Exception('Mixed data is not supported.')
        for var in varnames:
            candidate[var] = list(varnames)
            candidate[var].remove(var)
            dag[var] = {}
            dag[var]['par'] = []
            dag[var]['nei'] = []
            cache[var] = {}
            cache[var][tuple([])] = local_score(data, [varnames.index(var)], score_function)
        time_preprocess = time.time() - start_time
        diff = 1
        while diff > 0:
            diff = 0
            edge_candidate = []
            for vi in varnames:
                # attempt to add edges vi->vj
                for vj in candidate[vi]:
                    # calculate the time for checking cycles
                    start_time = time.time()
                    cyc_flag = check_cycle(vi, vj, dag)
                    time_checkcycle += time.time() - start_time
                    if not cyc_flag:
                        # calculate the time for computing the score
                        start_time = time.time()
                        par_sea = tuple(sorted(dag[vj]['par'] + [vi]))
                        if par_sea not in cache[vj]:
                            cols = [varnames.index(x) for x in (vj, ) + par_sea]
                            cache[vj][par_sea] = local_score(data, cols, score_function)
                        time_score += time.time() - start_time
                        diff_temp = cache[vj][par_sea] - cache[vj][tuple(dag[vj]['par'])]
                        if debug:
                            print(vi, vj, diff_temp, 'a')
                        if diff_temp - diff > 1e-10:
                            diff = diff_temp
                            edge_candidate = [vi, vj, 'a']
                for par_vi in dag[vi]['par']:
                    # attempt to reverse edges from vi<-par_vi to vi->par_vi
                    # calculate the time for checking cycles
                    start_time = time.time()
                    cyc_flag = check_cycle(vi, par_vi, dag)
                    time_checkcycle += time.time() - start_time
                    if not cyc_flag:
                        # calculate the time for computing the score
                        start_time = time.time()
                        par_sea_par_vi = tuple(sorted(dag[par_vi]['par'] + [vi]))
                        if par_sea_par_vi not in cache[par_vi]:
                            cols = [varnames.index(x) for x in (par_vi, ) + par_sea_par_vi]
                            cache[par_vi][par_sea_par_vi] = local_score(data, cols, score_function)
                        par_sea_vi = tuple([x for x in dag[vi]['par'] if x != par_vi])
                        if par_sea_vi not in cache[vi]:
                            cols = [varnames.index(x) for x in (vi, ) + par_sea_vi]
                            cache[vi][par_sea_vi] = local_score(data, cols, score_function)
                        time_score += time.time() - start_time
                        diff_temp = cache[par_vi][par_sea_par_vi] + cache[vi][par_sea_vi] - cache[par_vi][
                            tuple(dag[par_vi]['par'])] - cache[vi][tuple(dag[vi]['par'])]
                        if diff_temp - diff > 1e-10:
                            diff = diff_temp
                            edge_candidate = [vi, par_vi, 'r']
                        if debug:
                            print(vi, par_vi, diff_temp, 'r')

                    # attempt to delete edges vi<-par_vi
                    # calculate the time for computing the score
                    start_time = time.time()
                    par_sea = tuple([x for x in dag[vi]['par'] if x != par_vi])
                    if par_sea not in cache[vi]:
                        cols = [varnames.index(x) for x in (vi, ) + par_sea]
                        cache[vi][par_sea] = local_score(data, cols, score_function)
                    time_score += time.time() - start_time
                    diff_temp = cache[vi][par_sea] - cache[vi][tuple(dag[vi]['par'])]
                    if diff_temp - diff > 1e-10:
                        diff = diff_temp
                        edge_candidate = [par_vi, vi, 'd']
                    if debug:
                        print(par_vi, vi, diff_temp, 'd')
            if edge_candidate:
                if edge_candidate[-1] == 'a':
                    dag[edge_candidate[1]]['par'] = sorted(dag[edge_candidate[1]]['par'] + [edge_candidate[0]])
                    candidate[edge_candidate[0]].remove(edge_candidate[1])
                    candidate[edge_candidate[1]].remove(edge_candidate[0])
                elif edge_candidate[-1] == 'r':
                    dag[edge_candidate[1]]['par'] = sorted(dag[edge_candidate[1]]['par'] + [edge_candidate[0]])
                    dag[edge_candidate[0]]['par'].remove(edge_candidate[1])
                elif edge_candidate[-1] == 'd':
                    dag[edge_candidate[1]]['par'].remove(edge_candidate[0])
                    candidate[edge_candidate[0]].append(edge_candidate[1])
                    candidate[edge_candidate[1]].append(edge_candidate[0])
                if debug:
                    print('best operation is:', edge_candidate, diff)
        dag = bnlearn.model2network(to_bnlearn(dag))
        time_total = time.time() - time_total
        if debug:
            print('total cost:', time_total, 'seconds')
            print('preprocess cost: %.2f%%' % (time_preprocess / time_total * 100))
            print('check cycle cost: %.2f%%' % (time_checkcycle / time_total * 100))
            print('compute score cost: %.2f%%' % (time_score / time_total * 100))
            print('others cost: %.2f%%' % (
                        (time_total - time_preprocess - time_checkcycle - time_score) / time_total * 100))
    elif method == 'bnlearn':
        dag = bnlearn.hc(data, score=score_function, debug=debug)
    elif method == 'sem':
        dag = bnlearn.structural_em(data, maximize_args=ListVector({'score': score_function}), debug=debug)
    else:
        raise Exception('The input method: ' + method + ' is invalid.')
    if len(dc):
        modelstring = str(bnlearn.modelstring(dag))
        modelstring = modelstring[5:len(modelstring) - 2]
        for node in dc:
            modelstring += '[' + node + ']'
        dag = bnlearn.model2network(modelstring)
    return dag