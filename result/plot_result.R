setwd("~/Documents/GitHub/SFMV/result/")
library(ggplot2)
library(ggpubr)
rm(list = ls())
# result_em = read.csv('SEM.csv')
result_test = read.csv('test.csv')

result = data.frame(datasize = integer(), noise = character(), method = character(), F1 = numeric(), SHD = numeric())

for (datasize in unique(result_test$datasize)) {
  for (noise in c('MCAR', 'MAR', 'MNAR')) {
    result[nrow(result) + 1, ] = list(datasize, noise, 'HC-complete', mean(result_test[(result_test$datasize == datasize) & (result_test$noise == 'Clean'), 'F1']), mean(result_test[(result_test$datasize == datasize) & (result_test$noise == 'Clean'), 'SHD']))
    # result[nrow(result) + 1, ] = list(datasize, noise, 'Structural EM', mean(result_em[(result_em$datasize == datasize) & (result_em$noise == noise), 'F1']), mean(result_em[(result_em$datasize == datasize) & (result_em$noise == noise), 'SHD']))
    result[nrow(result) + 1, ] = list(datasize, noise, 'HC-pairwise', mean(result_test[(result_test$datasize == datasize) & (result_test$noise == noise) & (result_test$method == 'pw'), 'F1']), mean(result_test[(result_test$datasize == datasize) & (result_test$noise == noise) & (result_test$method == 'pw'), 'SHD']))
    result[nrow(result) + 1, ] = list(datasize, noise, 'HC-IPW', mean(result_test[(result_test$datasize == datasize) & (result_test$noise == noise) & (result_test$method == 'ipw'), 'F1']), mean(result_test[(result_test$datasize == datasize) & (result_test$noise == noise) & (result_test$method == 'ipw'), 'SHD']))
    result[nrow(result) + 1, ] = list(datasize, noise, 'HC-aIPW', mean(result_test[(result_test$datasize == datasize) & (result_test$noise == noise) & (result_test$method == 'aipw'), 'F1']), mean(result_test[(result_test$datasize == datasize) & (result_test$noise == noise) & (result_test$method == 'aipw'), 'SHD']))
    result[nrow(result) + 1, ] = list(datasize, noise, 'HC-nal', mean(result_test[(result_test$datasize == datasize) & (result_test$noise == noise) & (result_test$method == 'nal'), 'F1']), mean(result_test[(result_test$datasize == datasize) & (result_test$noise == noise) & (result_test$method == 'nal'), 'SHD']))
  }
}
result$noise = factor(result$noise, levels = c('MCAR', 'MAR', 'MNAR'))
result$method = factor(result$method, levels = c('HC-complete', 'HC-pairwise', 'HC-IPW', 'HC-aIPW', 'HC-nal'))
# result.m$datasize = as.factor(result.m$datasize)
# result.c$datasize = as.factor(result.c$datasize)
p.f1 = ggplot(data = result, aes(datasize, F1)) + geom_line(aes(color = method), size = 1) + geom_point(aes(color = method, shape = method), size = 2) + facet_grid(. ~ noise)+ theme(axis.title.x=element_blank(), axis.text.x = element_blank(), axis.ticks.x = element_blank())
p.shd = ggplot(data = result, aes(datasize, SHD)) + geom_line(aes(color = method), size = 1) + geom_point(aes(color = method, shape = method), size = 2) + facet_grid(. ~ noise)
plot(ggarrange(p.f1, p.shd, nrow = 2, common.legend = TRUE, legend = 'right'))