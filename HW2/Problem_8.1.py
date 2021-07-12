# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 18:42:50 2021

@author: User
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree
import sys

def build_model(df_train,model):
    train_x=[]
    df_importances = pd.DataFrame(columns=["Feature","Importance"])
    for i in range(0,200):
        if(len(df_train.columns)==11):
            train_x.append(np.array([df_train.loc[i,"V1"],df_train.loc[i,"V2"],df_train.loc[i,"V3"],df_train.loc[i,"V4"],df_train.loc[i,"V5"],df_train.loc[i,"V6"],df_train.loc[i,"V7"],df_train.loc[i,"V8"],df_train.loc[i,"V9"],df_train.loc[i,"V10"]]))
            df_importances.Feature=["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10"]
        else:
            train_x.append(np.array([df_train.loc[i,"V1"],df_train.loc[i,"V2"],df_train.loc[i,"V3"],df_train.loc[i,"V4"],df_train.loc[i,"V5"],df_train.loc[i,"V6"],df_train.loc[i,"V7"],df_train.loc[i,"V8"],df_train.loc[i,"V9"],df_train.loc[i,"V10"],df_train.loc[i,"V11"]]))
            df_importances.Feature=["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11"]
    train_y = df_train["y"]
    if(model=="RF"):
        rf = RandomForestRegressor(n_estimators = 200, random_state = 0)
        rf.fit(train_x, train_y)
        imp_out = rf.feature_importances_
    elif(model=="GBM"):
        rf = tree.DecisionTreeRegressor()
        rf.fit(train_x, train_y)
        imp_out = rf.feature_importances_
    elif(model=="DT"):
       model = GradientBoostingRegressor(max_depth=1)
       model.fit(train_x,train_y)
       imp_out=model.feature_importances_
    else:
         print("Not a Valid Model")
         sys.exit()
          
    imp_list=[]
    for i in range(0,len(imp_out)):
        imp_list.append(imp_out[i])

    df_importances.Importance=imp_list
    return(df_importances)



df_train = pd.read_csv("simulated.csv")
importance_p1 = build_model(df_train,"RF")
importance_p1

random_variable=list(np.random.normal(2, 4, 200)*0.1)
new_col = []
old_col = df_train["V1"]
for x in range(0,200):
    new_col.append(old_col[x]+random_variable[x])

df_train["V11"] = new_col

importance_p2 = build_model(df_train,"RF")
importance_p2

importance_gbm = build_model(df_train,"GBM")
importance_gbm 

importance_dt = build_model(df_train,"DT")
importance_dt 