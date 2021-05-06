# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 17:51:45 2021

@author: renji
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
"%matplotlib inline"


# Machine_Learning

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

train_data = pd.read_csv('D:\\TCE\\sem 6\\Data Science using Python\\Predict price\\train.csv') 
test_data = pd.read_csv('D:\\TCE\\sem 6\\Data Science using Python\\Predict price\\train.csv')


x_train = train_data.drop('price_range',axis=1)
y_train = train_data['price_range']
test_data = test_data.drop('price_range', axis=1)

X_train, X_val, Y_train, Y_val = train_test_split(x_train,y_train, test_size= 0.2, random_state= 5)

X_train.shape, Y_train.shape, X_val.shape, Y_val.shape

#logistic regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(test_data)
acc_log = round(logreg.score(X_val, Y_val) * 100, 2)
acc_log
print(acc_log)

coeff_df = pd.DataFrame(train_data.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)

#SVC

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(test_data)
acc_svc = round(svc.score(X_val, Y_val) * 100, 2)
acc_svc
print(acc_svc)

#random forest


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(test_data)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_val, Y_val) * 100, 2)
acc_random_forest
print(acc_random_forest)

# KNeighborsClassifier

#knn = KNeighborsClassifier(n_neighbors = 3)
#knn.fit(X_train, Y_train)
#Y_pred = knn.predict(test_data)
# = round(knn.score(X_val, Y_val) * 100, 2)
#acc_knn

# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(test_data)
acc_gaussian = round(gaussian.score(X_val, Y_val) * 100, 2)
acc_gaussian
print(acc_gaussian)

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes'],
    'Accuracy': [acc_svc, acc_log, 
              acc_random_forest, acc_gaussian]})
models.sort_values(by='Accuracy', ascending=False)
frame=pd.DataFrame(models)

sns.barplot(y='Model',x='Accuracy',data=frame.sort_values(by = 'Accuracy',
                                                              ascending = False))

test_data.head()

predicted_price_range = svc.predict(test_data)

predicted_price_range

test_data['price_range'] = predicted_price_range

test_data.head()

