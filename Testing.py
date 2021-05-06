# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 20:49:21 2021

@author: renji
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
data_csv = pd.read_csv('D:\\TCE\\sem 6\\Data Science using Python\\Predict price\\train.csv',na_values='?')
#examination of sampling distribution
data=data_csv['price_range'].copy()
plt.hist(data,bins=50,density=True)


#Finding the log-likelihood of the samples using score_samples function
x_value=np.linspace(1,2,100)
conf=KernelDensity(bandwidth=0.007)
conf.fit(data[:,np.newaxis])
log_likelihood=conf.score_samples(x_value[:,np.newaxis])
log_score=np.exp(log_likelihood)
plt.plot(x_value,log_score,alpha=0.9,lw=5,color='g')
plt.xlabel("price_range")
plt.ylabel("Density")
plt.title("Examination of sampling distribution")
plt.show()


#Finding mean,standard deviation of the population
mean=np.mean(data)
std=np.std(data)
print(""" Mean of the population is : %.3f
Standard Deviation of population : %.3f
Total size of the population is : %i
"""%(mean,std,len(data)))

#Sampling the population and identifying the sample mean and standard deviation
sample_size=5000
notrials=50000
sample=np.array([np.random.choice(data,sample_size) for i in range(notrials)])
smean=sample.mean(axis=1)
sstd=np.std(smean)
sam_mean=np.mean(smean)
ana_std=std/np.sqrt(sample_size)
print("""Mean of the sample : %.3f
Standard Deviation of the sample : %.3f
Analytical std: %.3f
"""%(sam_mean,std,ana_std))


from scipy.stats import t
#Finding the lower tail and the upper tail of the sampling distribution
z=1.96
plt.hist(smean,bins=50,density=True)
plt.axvline(sam_mean-1.96*sstd,color='r')
plt.axvline(sam_mean+1.96*sstd,color='r')
plt.xlabel("Mean of the sample")
plt.ylabel("Density")
plt.title("Lower and upper tail of sample")
plt.show()
lower=100*sum(smean<sam_mean-1.96*sstd)/len(smean)
upper=100*sum(smean>sam_mean+1.96*sstd)/len(smean)
print("Lower tail of the sample:%.3f%%"%(lower))
print("Upper tail of the sample:%.3f%%"%(upper))


import pylab
import scipy.stats as stats
#plotting of QQ plot
stats.probplot(smean,dist="norm",plot=pylab)
pylab.show()


#finding false positive rate
z=1.96
se=sample.std(axis=1)
ana_mean=se/np.sqrt(sample_size)
upper_tail=smean+z*ana_mean
lower_tail=smean-z*ana_mean
positive=np.mean((mean>=lower_tail)&(mean<=upper_tail))
fpr=np.mean((mean<lower_tail)|(mean>upper_tail))
print("False positive rate is : %.3f"%fpr)


#plotting confidence interval
npoints=9000
plt.scatter(list(range(len(upper_tail[:npoints]))),upper_tail[:npoints])
plt.scatter(list(range(len(lower_tail[:npoints]))),lower_tail[:npoints])
plt.axhline(y=0.0551)
plt.xlabel("Sample Distribution")
plt.ylabel("Mean of the sample")
plt.title("Confidence Interval")
plt.show()


from scipy.stats import anderson
data=data_csv["price_range"].copy()
test_statistic=anderson(data)
print('stat=%.3f'%(test_statistic.statistic))
#Comparing the test statistic value to each critical value that corresponds to each significance
#level to test if the test results are significant
for i in range(len(test_statistic.critical_values)):
    sl,cv=test_statistic.significance_level[i],test_statistic.critical_values[i]
    if test_statistic.statistic<cv:
        print("Data follows normal at %.1f%% level"%(sl))
    else:
        print("Data does not follow normal at %.1f%% level"%(sl))

#Correlation Test-Spearman's Rank Correlation
from scipy.stats import spearmanr
data1=data_csv['ram'].copy()
data2=data_csv['n_cores'].copy()
res,p=spearmanr(data1,data2)
print('stat=%.3f,p=%.3f'%(res,p))
if p>0.05:
    print('Samples are correlated')
else:
    print('Samples are not correlated')
    
#Stationary test-Augmented Dickey Fuller unit Root test
from statsmodels.tsa.stattools import adfuller
data=data_csv['clock_speed'].copy()
stat,p,lags,obs,crit,t=adfuller(data)
print('stat=%.3f,p=%.3f'%(stat,p))
if p>0.05:
    print(' Sample are Not Stationary')
else:
    print('Samples are Stationary')
    
#ANOVA test:One way analysis
from scipy.stats import f_oneway
data1=data_csv['ram']
data2=data_csv['n_cores']
data3=data_csv['battery_power']
F,p=f_oneway(data1,data2,data3)
print('stat=%.3f,p=%.3f'%(F,p))
if p>0.05:
    print('Samples are probably have same distribution')
else:
    print('Samples are probably have different distribution')
    
#Student T test
from scipy.stats import ttest_ind
data1=data_csv['ram']
data2=data_csv['n_cores']
F,p=ttest_ind(data1,data2)
print('stat=%.3f,p=%.3f'%(F,p))
if p>0.05:
    print("Probably in same distribution")
else:
    print("Probably in different distribution")
    
#sample z test
from scipy import stats
from statsmodels.stats import weightstats as stests
ztest,pval=stests.ztest(data_csv['ram'],x2=data_csv['n_cores'],
value=0,alternative='two-sided')
print("stat=%.3f,p=%.3f"%(ztest,pval))
if pval>0.05:
    print("Accept null hypothesis")
else:
    print("reject null hypothesis")
    

#creating equal sized intervals and creating a dataframe
p=np.linspace(-5,5,20)
sin_p=np.sin(p)
cos_p=np.cos(p)
pd.DataFrame({'t':t,'sin':sin_p,'cos':cos_p})


#grouping the data based on price_range
groupby_class=data_csv.groupby('price_range')
for price_range,value in groupby_class['ram']:
    print((price_range,value.mean()))
groupby_class.mean()


#plotting a scatter matrix
from pandas.plotting import scatter_matrix
scatter_matrix(data_csv[['ram','n_cores','int_memory']])


scatter_matrix(data_csv[['n_cores','int_memory','ram','clock_speed']])