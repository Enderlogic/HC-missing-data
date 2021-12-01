setwd("~/Documents/GitHub/HC-missing-data/result/")
library(ggplot2)
library(ggpubr)
rm(list = ls())

result_test = read.csv('sparse_result.csv')

result = data.frame(datasize = integer(), noise = character(), method = character(), F1 = numeric(), SHD = numeric())

for (datasize in unique(result_test$datasize)) {
  for (noise in c('MCAR', 'MAR', 'MNAR')) {
    result[nrow(result) + 1, ] = list(datasize, noise, 'HC-complete', mean(result_test[(result_test$datasize == datasize) & (result_test$noise == 'Clean'), 'F1']), mean(result_test[(result_test$datasize == datasize) & (result_test$noise == 'Clean'), 'SHD']))
    # result[nrow(result) + 1, ] = list(datasize, noise, 'Structural EM', mean(result_test[(result_test$datasize == datasize) & (result_test$noise == noise) & (result_test$method == 'sem'), 'F1']), mean(result_test[(result_test$datasize == datasize) & (result_test$noise == noise) & (result_test$method == 'sem'), 'SHD']))
    result[nrow(result) + 1, ] = list(datasize, noise, 'HC-pairwise', mean(result_test[(result_test$datasize == datasize) & (result_test$noise == noise) & (result_test$method == 'pw'), 'F1']), mean(result_test[(result_test$datasize == datasize) & (result_test$noise == noise) & (result_test$method == 'pw'), 'SHD']))
    result[nrow(result) + 1, ] = list(datasize, noise, 'HC-IPW', mean(result_test[(result_test$datasize == datasize) & (result_test$noise == noise) & (result_test$method == 'ipw'), 'F1']), mean(result_test[(result_test$datasize == datasize) & (result_test$noise == noise) & (result_test$method == 'ipw'), 'SHD']))
    result[nrow(result) + 1, ] = list(datasize, noise, 'HC-aIPW', mean(result_test[(result_test$datasize == datasize) & (result_test$noise == noise) & (result_test$method == 'aipw'), 'F1']), mean(result_test[(result_test$datasize == datasize) & (result_test$noise == noise) & (result_test$method == 'aipw'), 'SHD']))
  }
}
result$noise = factor(result$noise, levels = c('MCAR', 'MAR', 'MNAR'))
# result$method = factor(result$method, levels = c('HC-pairwise', 'HC-IPW', 'HC-aIPW', 'Structural EM', 'HC-complete'))
result$method = factor(result$method, levels = c('HC-pairwise', 'HC-IPW', 'HC-aIPW', 'HC-complete'))
result$datasize = factor(result$datasize)
p.f1 = ggplot(data = result, aes(datasize, F1, group=method)) + geom_line(aes(color = method, linetype=method), size = 1) + geom_point(aes(color = method, shape = method), size = 2) + scale_linetype_manual(values=c("solid", "solid", "solid", "solid", "dashed")) + facet_grid(. ~ noise)+ theme(axis.title.x=element_blank(), axis.text.x = element_blank(), axis.ticks.x = element_blank())
p.shd = ggplot(data = result, aes(datasize, SHD, group=method)) + geom_line(aes(color = method, linetype=method), size = 1) + geom_point(aes(color = method, shape = method), size = 2) + scale_linetype_manual(values=c("solid", "solid", "solid", "solid", "dashed")) + facet_grid(. ~ noise)
plot(ggarrange(p.f1, p.shd, nrow = 2, common.legend = TRUE, legend = 'right'))