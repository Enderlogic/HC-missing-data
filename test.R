setwd("~/Documents/GitHub/SFMV")
library(bnlearn)
rm(list = ls())
data = read.csv('data/1/1_Clean.csv')
data = as.data.frame(lapply(data, as.factor))

data.mcar = read.csv('data/1/1_MCAR.csv', na.strings = c('NaN', ''))
data.mcar = as.data.frame(lapply(data.mcar, as.factor))

data.mar = read.csv('data/1/1_MAR.csv', na.strings = c('NaN', ''))
data.mar = as.data.frame(lapply(data.mar, as.factor))

local_nal = function(data, var, par=c()) {
  ct = table(data[, c(var, par)])
  
  ll = 1 / sum(ct) * sum(ct * log(prop.table(ct, par)), na.rm = TRUE)
  # nal = ll - 1/length(data) * sum(ct) ** -0.3 * (nlevels(data[, var]) - 1) * prod(unlist(lapply(data[par], nlevels)))
  nal = ll - 0.5 * log(sum(ct)) / sum(ct) * (nlevels(data[, var]) - 1) * prod(unlist(lapply(data[par], nlevels)))
  return(nal)
}
