library(bnlearn)
library(stringi)
library(rlist)

list <- structure(NA,class="result")
"[<-.result" <- function(x,...,value) {
  args <- as.list(match.call())
  args <- args[-c(1:2,length(args))]
  length(value) <- length(args)
  for(i in seq(along=args)) {
    a <- args[[i]]
    if(!missing(a)) eval.parent(substitute(a <- v,list(a=a,v=value[[i]])))
  }
  x
}

vst = function(dag) {
  for (node in nodes(dag)) {
    if (length(parents(dag, node)) > 1) {
      parents = parents(dag, node)
      for (parent in parents(dag, node)) {
        other_parents = setdiff(parents, parent)
        for (other_parent in other_parents) {
          if (!parent %in% nbr(dag, other_parent)) {
            if (exists('vstruc')) {
              vstruc = rbind(vstruc, c(parent, node, other_parent)) 
            }
            else {
              vstruc = matrix(c(parent, node, other_parent), ncol = 3)
              colnames(vstruc) = c('X', 'Z', 'Y')
            }
          }
        }
        parents = setdiff(parents, parent)
      }
    }
  }
  return(vstruc)
}

generate = function(datainfo, visual = FALSE, savemodel = FALSE, path = NULL) {
  # generate synthetic model given datainfo
  # datainfo contains: "datatype" (character): discrete or continuous;
  # "min.nodes" (numeric): minimum number of nodes;
  # "max.nodes" (numeric): maximum number of nodes;
  # for discrete model:
  # max.state (numeric): maximum number of states
  # for continuous model:
  # gaussian (logical): gaussian model or not
  # func (character): linear or manually defined function (not implemented yet)
  
  # visual: plot the BN if it is TRUE
  # savemodel: save the model or not;
  # path: the path to save model
  if (datainfo$max.nodes == datainfo$min.nodes)
    # node.names = as.character(1 : datainfo$max.nodes)
    node.names = stri_rand_strings(datainfo$max.nodes, 5, pattern = '[A-Za-z]')
  else
    # node.names = as.character(1 : sample(datainfo$min.nodes : datainfo$max.nodes, 1))
    node.names = stri_rand_strings(sample(datainfo$min.nodes : datainfo$max.nodes, 1), 5, pattern = '[A-Za-z]')
  dag = random.graph(node.names, prob = 2 / (length(node.names) - 1))
  
  dist = vector(mode = 'list', length = length(node.names))
  names(dist) = node.names
  if (datainfo$continuous) {
    # continuous model
    if (datainfo$gaussian & datainfo$func == 'linear') {
      for (node in node.names) {
        coef = runif(length(dag$nodes[[node]]$parents), 0.1, 1) * sample(c(1, -1), length(dag$nodes[[node]]$parents), TRUE)
        coef = c(rnorm(1, sd =2), coef)
        names(coef) = c("(Intercept)", dag$nodes[[node]]$parents)
        dist[[node]] = list(coef = coef, sd = 1)
      }
      dag.fit = custom.fit(dag, dist = dist)
    } else {
      stop("Non-gaussian or non-linear models are currently not available.")
    }
  } else {
    # discrete model
    dims = sample(2 : datainfo$max.state, length(nodes(dag)), replace = TRUE)
    names(dims) = nodes(dag)
    for (node in nodes(dag)) {
      number.of.states = dims[[node]]
      if (length(dag$nodes[[node]]$parents) == 0) {
        dist[[node]] = matrix(rdirichlet(1, rep(1, number.of.states)), ncol = number.of.states, dimnames = list(NULL, LETTERS[1 : number.of.states]))
      } else {
        dimnames.temp = list(LETTERS[1 : number.of.states])
        dim.temp = c(number.of.states)
        for (parent in dag$nodes[[node]]$parents) {
          dim.temp = c(dim.temp, dims[[parent]])
          dimnames.temp = list.append(dimnames.temp, LETTERS[1 : dims[[parent]]])
        }
        names(dimnames.temp) = c(node, dag$nodes[[node]]$parents)
        dist[[node]] = t(rdirichlet(prod(dim.temp) / dim.temp[1], rep(1, number.of.states)))
        
        dim(dist[[node]]) = dim.temp
        dimnames(dist[[node]]) = dimnames.temp
      }
    }
    dag.fit = custom.fit(dag, dist = dist)
  }
  
  if (visual)
    graphviz.plot(dag.fit)
  if (savemodel) {
    dir.create(file.path(dirname(path)), showWarnings = FALSE, recursive = TRUE)
    saveRDS(dag.fit, path)
  }
  return(dag.fit)
}

add.mechanism = function(dag, p, noisetype, cause.max = 3) {
  var.miss = sample(nodes(dag), round(length(nodes(dag)) * p))
  var.comp = setdiff(nodes(dag), var.miss)
  cause.list = vector(mode = 'list', length = length(var.miss))
  names(cause.list) = var.miss
  if (noisetype != 'MCAR') {
    for (var in var.miss) {
      cause.num = sample(1: cause.max, 1)
      if (noisetype == 'MAR') {
        cause = sample(var.comp, cause.num)
      }
      else if (noisetype == 'MNAR') {
        var.else = setdiff(nodes(dag), var)
        cause = sample(var.else, cause.num)
      }
      else
        stop('Noise type must be in MCAR, MAR or MNAR')
      cause.list[[var]] = cause
    }
  }
  return(cause.list)
}

add.mechanism2 = function(dag, p, noise) {
  if (noise == 'MCAR') {
    cause.list = vector(mode = 'list', length = round(length(nodes(dag)) * p))
    var.miss = sample(nodes(dag), round(length(nodes(dag)) * p))
    names(cause.list) = var.miss
  } else if (noise == 'MAR') {
    cause.list = vector(mode = 'list')
    vst = vst(dag)
    nom = round(length(nodes(dag)) * p)
    noc = length(nodes(dag)) - nom
    vars_miss = c()
    vars_comp = c()
    for (i in 1 : nrow(vst)) {
      if ((length(vars_comp) != noc) | ((length(vars_comp) == noc) & (vst[i, 2] %in% vars_comp))) {
        if (!vst[i, 1] %in% c(vars_miss, vars_comp)) {
          cause.list[vst[i, 1]] = vst[i, 2]
          vars_miss = c(vars_miss, vst[i, 1])
          vars_comp = unique(c(vars_comp, vst[i, 2]))
        } else if (!vst[i, 3] %in% c(vars_miss, vars_comp)) {
          cause.list[vst[i, 3]] = vst[i, 2]
          vars_miss = c(vars_miss, vst[i, 3])
          vars_comp = unique(c(vars_comp, vst[i, 2]))
        }
        if (length(vars_miss) == nom)
          break
      }
    }
    if (length(vars_miss) < nom) {
      vars_miss2 = sample(setdiff(setdiff(nodes(dag), vars_miss), vars_comp), nom - length(vars_miss))
      vars_comp = setdiff(setdiff(nodes(dag), vars_miss), vars_miss2)
      for (var in vars_miss2) {
        if (length(intersect(nbr(dag, var), vars_comp)))
          cause.list[var] = sample(intersect(nbr(dag, var), vars_comp), 1)
        else
          cause.list[var] = sample(vars_comp, 1)
      }
    }
  } else if (noise == 'MNAR') {
    cause.list = vector(mode = 'list')
    vst = vst(dag)
    nom = ceiling(length(nodes(dag)) * p / 2)
    for (i in 1 : nrow(vst)) {
      if (!vst[i, 1] %in% names(cause.list))
        cause.list[vst[i, 1]] = vst[i, 2]
      else if (!vst[i, 3] %in% names(cause.list))
        cause.list[vst[i, 3]] = vst[i, 2]
      if (length(cause.list) == nom)
        break
    }
    if (length(cause.list) < nom) {
      vars_miss = sample(setdiff(nodes(dag), names(cause.list)), nom - length(cause.list))
      for (var in vars_miss) {
        if (length(nbr(dag, var)))
          cause.list[var] = sample(nbr(dag, var), 1)
        else
          cause.list[var] = sample(setdiff(nodes(dag), var), 1)
      }
    }
    vars_miss = setdiff(unlist(cause.list, use.names = FALSE), names(cause.list))
    if (length(unique(c(names(cause.list), unlist(cause.list, use.names = FALSE)))) < nom * 2)
      vars_miss = c(vars_miss, sample(setdiff(nodes(dag), unique(c(names(cause.list), unlist(cause.list, use.names = FALSE)))), 2 * nom - length(unique(c(names(cause.list), unlist(cause.list, use.names = FALSE))))))
    vars_comp = setdiff(setdiff(nodes(dag), vars_miss), unique(c(names(cause.list), unlist(cause.list, use.names = FALSE))))
    for (var in vars_miss) {
      if (length(intersect(nbr(dag, var), vars_comp)))
        cause.list[var] = sample(intersect(nbr(dag, var), vars_comp), 1)
      else
        cause.list[var] = sample(vars_comp, 1)
    }
  } else
    stop(paste('noise', noise, 'is undefined.'))
  return(cause.list)
}

add.mechanism3 = function(dag, p, noisetype) {
  var.miss = sample(nodes(dag), round(length(nodes(dag)) * p))
  var.comp = setdiff(nodes(dag), var.miss)
  cause.list = vector(mode = 'list', length = length(var.miss))
  names(cause.list) = var.miss
  if (noisetype != 'MCAR') {
    for (var in var.miss) {
      if (noisetype == 'MAR')
        cause = sample(var.comp, 1)
      else if (noisetype == 'MNAR') {
        if (runif(1) > 0.5)
          cause = sample(setdiff(var.miss, var), 1)
        else
          cause = sample(var.comp, 1)
      }
      else
        stop('Noise type must be in MCAR, MAR or MNAR')
      cause.list[[var]] = cause
    }
  }
  return(cause.list)
}

add.missing = function(data, cause.list = NULL, m.min = 0.1, m.max = 0.6) {
  # add missing value to the input dataset
  # inputs:
  # data: dataset without missing values
  # cause.list: list of causes of missing indicators
  # m.min: minimal proportion of missing value for each variable in cause.list
  # m.max: maximal proportion of missing value for each variable in cause.list
  # output:
  # data.with.noise: dataset with missing values
  if (m.max < m.min | m.max > 1 | m.min < 0)
    stop("The input minimal or maximal proportion of missing value are invalid")
  data.with.noise = data
  for (var in names(cause.list)) {
    cause = cause.list[[var]]
    if (length(cause) == 0)
      data.with.noise[[var]][runif(nrow(data)) < runif(1, min = m.min, max = m.max)] = NA
    else {
      missing.indicator = rep(FALSE, nrow(data))
      if (is.factor(data[[var]])) {
        for (i in 1 : length(cause)) {
          state.m = tail(names(sort(table(data[cause[i]]))), 1)
          # state.m = sample(levels(data[[cause[i]]]), 1)
          state.others = setdiff(levels(data[[cause[i]]]), state.m)
          missing.indicator = missing.indicator | (data[[cause[i]]] %in% state.m & runif(nrow(data)) < m.max)
          missing.indicator = missing.indicator | (data[[cause[i]]] %in% state.others & runif(nrow(data)) < m.min)
        }
      } else {
        m = runif(length(cause), min = m.min, max = m.max)
        if (length(cause) > 1)
          thres = diag(apply(data[cause], 2, quantile, probs = m))
        else
          thres = quantile(data[[cause]], m)
        for (i in 1 : length(cause)) {
          missing.indicator = missing.indicator | (data[cause[i]] < thres[i] & runif(nrow(data)) < m.max)
          missing.indicator = missing.indicator | (data[cause[i]] >= thres[i] & runif(nrow(data)) < m.min)
        }
      }
      data.with.noise[[var]][missing.indicator] = NA
    }
  }
  return(data.with.noise)
}