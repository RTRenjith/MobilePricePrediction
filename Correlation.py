# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 09:48:43 2021

@author: renji
"""

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

path ='D:\\TCE\\sem 6\\Data Science using Python\\Predict price\\train.csv'
data_train = pd.read_csv(path)
data_train.head()

data_train.info()

fig = plt.figure(figsize =(12,10))
np.set_printoptions(precision=2)
sns.heatmap(data_train.corr(), annot = True, fmt=".2f")

sns.pairplot(data_train[['price_range','ram']])