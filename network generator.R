library(stringi)
library(bnlearn)
library(gtools)
library(rlist)
setwd("/Users/yangliu/Documents/GitHub/HC-missing-data/")
rm(list = ls())
set.seed(1234)
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
    node.names = stri_rand_strings(sample(datainfo$min.nodes : datainfo$max.nodes, 1), 5, pattern = '[A-Za-z]')
  # adj = matrix(0, ncol = length(node.names), nrow = length(node.names), dimnames = list(node.names, node.names))
  # for (i in seq(2, length(node.names))) {
  #   parents = sample(node.names[1 : i-1], sample(seq(0, min(i-1, 3)), 1))
  #   for (par in parents)
  #     adj[par, node.names[i]] = 1
  # }
  # dag = empty.graph(node.names)
  # amat(dag) = adj
  dag = random.graph(node.names, prob = 2 / (length(node.names) - 1))
  
  dist = vector(mode = 'list', length = length(node.names))
  names(dist) = node.names
  if (datainfo$datatype == 'continuous') {
    # continuous model
    if (datainfo$gaussian & datainfo$func == 'linear') {
      for (node in node.names) {
        coef = runif(length(dag$nodes[[node]]$parents), 0.1, 1) * sample(c(1, -1), length(dag$nodes[[node]]$parents), TRUE)
        coef = c(rnorm(1, sd =4), coef)
        names(coef) = c("(Intercept)", dag$nodes[[node]]$parents)
        # dist[[node]] = list(coef = coef, sd = runif(1, 0.1, 4))
        dist[[node]] = list(coef = coef, sd = 1)
      }
      dag.fit = custom.fit(dag, dist = dist)
    } else {
      stop("Non-gaussian or non-linear models are currently not available.")
    }
  } else if (datainfo$datatype == 'discrete') {
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
  } else
    stop('The type of data is invalid')
  
  if (visual)
    graphviz.plot(dag.fit)
  if (savemodel) {
    dir.create(file.path(dirname(path)), showWarnings = FALSE, recursive = TRUE)
    saveRDS(dag.fit, path)
  }
  return(dag.fit)
}

datainfo = vector(mode = 'list')
datainfo = c(datainfo, datatype = 'discrete') # set the type of data (continuous or discrete)

datainfo = c(datainfo, max.nodes = 50) # set the maximum number of nodes
datainfo = c(datainfo, min.nodes = 20)
if (datainfo$datatype == 'discrete') {
  datainfo = c(datainfo, max.state = 6) # set the maximum number of state if variable is discrete 
} else if (datainfo$datatype == 'continuous') {
  datainfo = c(datainfo, gaussian = TRUE)
  datainfo = c(datainfo, func = 'linear')
} else
  stop('invalid datatype')
num.of.model = 50
model.list = as.character(1 : num.of.model)

average.degree = c()
for (i in 1 : num.of.model) {
  path = paste0('sparse_network/', i, '.rds')
  dag = generate(datainfo, visual = FALSE, savemodel = TRUE, path = path)
  ad = 0
  for (var in nodes(dag))
    ad = ad + degree(dag, var)
  average.degree = c(average.degree, ad / length(dag))
}
print(mean(average.degree))