# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 22:31:42 2021

@author: User
"""

import random
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from matplotlib import pyplot

random.seed(755)

x = np.array([1,2])
X_low=list(np.repeat(x, 300, axis=0))
X_low
X_high=list(np.random.normal(2, 4, 600))
X_high
X = list()
Y = list()

for x in range(0,600):
    Y.append(X_low[x] + X_high[x])
    X.append(np.array([X_low[x],X_high[x]]))

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, Y)

# get importance
importance = clf.feature_importances_
# summarize feature importance

for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()