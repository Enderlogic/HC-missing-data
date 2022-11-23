# Introduction
This is the source code of the paper [Greedy structure learning from data that contain systematic missing values](https://link.springer.com/article/10.1007/s10994-022-06195-8) which aims to solve missing value problems for the Hill-Climbing algorithm. We provide four methods to handle missing values which are list-wise deletion (lw), pair-wise deletion (pw), inverse probability weighting (ipw) and adaptive inverse probability weighting (aipw).

If you want to test our algorithm on the data set in your interest, please put your data and network in the "data" and "networks" folder respectively, and change the corresponding variables in "main.py" file which is also an example file for running the code. The format of the networks should be the same as the RDS files provided in the bnlearn repository, for more information please check https://www.bnlearn.com/bnrepository/.

If you use our code, please consider citing:

```
@article{liu2022greedy,
  title={Greedy structure learning from data that contain systematic missing values},
  author={Liu, Yang and Constantinou, Anthony C},
  journal={Machine Learning},
  volume={111},
  number={10},
  pages={3867--3896},
  year={2022},
  publisher={Springer}
}
```
