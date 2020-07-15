import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.plotting import scatter_matrix
from sklearn.metrics import accuracy_score
import os
import ast

def read_dataset(path):
    return pd.read_csv(path)
df_train = read_dataset('data_frame.csv')
df_train.columns
X = df_train.iloc[:,1:].values
y = df_train.iloc[:,0].values

#split training and testing sets in 90:10 ratio
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1) 

#use a random forest classifier to train the data
model = RandomForestClassifier(max_depth=20,random_state=40)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test,y_pred)

y_pred=model.predict(X_test[7].reshape(1,-1))
y_pred,X_test[4]

img = plt.imread("map_entc.jpeg")
number=y_pred
y_cart=np.array([[40,33,1],[12,34,2],[20,22,3],[30,28,4],[51,27,5],[63,14,6],[44,18,7],[80,31,8],[77,28,9],[60,28,10],[63,34,11],[85,35,12],[30,32,13],[62,17,14],[37,28,15],[55,26,16],[47,32,17],[52,18,18],[56,12,19],[34,18,20],[59,20,21],[70,37,22],[67,30,23]])
for i in range(len(y_cart)):
    if (y_cart[i][2]==number):
        plt.imshow(img,extent=[0,100,0,60])
        plt.scatter(y_cart[i][0],y_cart[i][1],marker='x',color='blue')
        plt.text(y_cart[i][0],y_cart[i][1], y_cart[i][2], fontsize=10)
    else:
        plt.scatter(y_cart[i][0],y_cart[i][1],color='red')
        plt.text(y_cart[i][0],y_cart[i][1], y_cart[i][2], fontsize=10)
