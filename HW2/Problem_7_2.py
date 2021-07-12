# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 17:24:17 2021

@author: User
"""

import pandas as pd
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from pyearth import Earth
from sklearn.model_selection import cross_val_score



df_train = pd.read_csv("trainingData.csv")

df_train




train_x=[]
for i in range(0,200):
    train_x.append(np.array([df_train.loc[i,"x.X1"],df_train.loc[i,"x.X2"],df_train.loc[i,"x.X3"],df_train.loc[i,"x.X4"],df_train.loc[i,"x.X5"],df_train.loc[i,"x.X6"],df_train.loc[i,"x.X7"],df_train.loc[i,"x.X8"],df_train.loc[i,"x.X9"],df_train.loc[i,"x.X10"]]))

train_y = df_train["y"]

len(train_x)
len(train_y)

def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

regressor = SVR(kernel='linear')
regressor.fit(train_x,train_y)
reg_train = regressor.predict(train_x)




print("SVR Train RMSE: %.2f"
      % np.sqrt(mean_squared_error(train_y, reg_train)))

regressor.coef_[0]
out=f_importances(regressor.coef_[0], ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10'])


model = Earth(feature_importance_type=('rss', 'gcv', 'nb_subsets'))
model.fit(train_x,train_y)

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, train_x, train_y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

rmse_cv(model).mean()


print(model.trace())

print(model.summary_feature_importances(sort_by='gcv'))
