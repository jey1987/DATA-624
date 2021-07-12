# gradient boosting for regression in scikit-learn
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib import pyplot
import pandas as pd


X=pd.read_csv('solTrainXtrans.csv') 
y=pd.read_csv('solTrainY.csv') 
X_matrix = np.array(X)
y = np.array(y)
X_matrix
len(y)
X
y
model = GradientBoostingRegressor(max_depth=1)
model.fit(X_matrix, y)


# get importance
importance = model.feature_importances_
# summarize feature importance

df = pd.DataFrame(columns=['feature','importance'])
cols=X.columns[1:229]
cols
rows_list = []
for i,v in enumerate(importance):
    dic1={}
    feature_res=cols[i-1]
    importance_res=v
    dic1.update({'feature':feature_res,'importance':v})
    rows_list.append(dic1)
df=df.append(rows_list)

    
df_result_1=df.sort_values(by='importance', ascending=False).head(10)


model = GradientBoostingRegressor(max_depth=10)
model.fit(X_matrix, y)


# get importance
importance = model.feature_importances_
# summarize feature importance

df = pd.DataFrame(columns=['feature','importance'])
cols=X.columns[1:229]
cols
rows_list = []
for i,v in enumerate(importance):
    dic1={}
    feature_res=cols[i-1]
    importance_res=v
    dic1.update({'feature':feature_res,'importance':v})
    rows_list.append(dic1)
df=df.append(rows_list)
    
df_result_10=df.sort_values(by='importance', ascending=False).head(10)


df_result_1    
df_result_10
