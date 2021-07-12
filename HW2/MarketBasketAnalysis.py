import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
import seaborn as sns

df = pd.read_csv("GroceryDataSet.csv")

transactions = []
for i in range(0, 9834):
    transactions.append([str(df.values[i,j]) for j in range(0, 23)])



df = pd.DataFrame({'freq': sum(transactions,[])})
df_freq=df.groupby('freq', as_index=False).size().sort_values(by="size",ascending=False).head(20).dropna()
df_freq=df_freq[df_freq['freq']!='nan']

df_freq.plot.bar(x='freq',y='size')

plt.show() 
    
rule_list = apriori(transactions, min_support = 0.004, min_confidence = 0.3)

results = list(rule_list)
df_results  = pd.DataFrame(results)
df_results.head()
support = df_results.support
len(results)

first_values = []
second_values = []
third_values = []
fourth_value = []

for i in range(df_results.shape[0]):
    single_list = df_results['ordered_statistics'][i][0]
    first_values.append(list(single_list[0]))
    second_values.append(list(single_list[1]))
    third_values.append(single_list[2])
    fourth_value.append(single_list[3])
    
lhs = pd.DataFrame(first_values)
rhs= pd.DataFrame(second_values)
confidence=pd.DataFrame(third_values,columns=['confidence'])
lift=pd.DataFrame(fourth_value,columns=['lift'])

df_final = pd.concat([lhs,rhs,support,confidence,lift], axis=1)
df_final.sort_values(by='lift', ascending=False).head(10)

scatter = sns.scatterplot(x="support", y="confidence",
                            hue="lift", # color dots by lift value
                            palette="viridis", # set colors
                            data=df_final)



parallel = pd.plotting.parallel_coordinates(
               df_final,
               "support", # set column containing rule number
               colormap='viridis', # set color palette
               sort_labels=True)