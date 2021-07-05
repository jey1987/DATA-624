library(arules)
library(RColorBrewer)

df <- read.csv("GroceryDataSet.csv")

df_sparse <- read.transactions("GroceryDataset.csv",format="basket",sep=",")
summary(df_sparse)


itemFrequencyPlot(df_sparse,topN=20,type="absolute",col=brewer.pal(8,'Pastel2'), main="Frequently Purchased Products")

association.rules <- apriori(df_sparse, parameter = list(supp=0.004, conf=0.3))

length(association.rules)
inspect(sort(association.rules, by = 'lift')[1:10])



subset.rules <- which(colSums(is.subset(association.rules, association.rules)) > 1) # get subset rules in vector
length(subset.rules)  


subset.association.rules. <- association.rules[-subset.rules] # remove subset rules.


inspect(sort(subset.association.rules., by = 'lift')[1:10])

library(arulesViz)
library(visNetwork)
library(igraph)

plot(subset.association.rules.,method="two-key plot")
plot(association.rules)


top10subRules <- head(subset.association.rules., n = 10, by = "lift")
plot(top10subRules, method = "graph",  engine = "htmlwidget")

subRules2<-head(subset.association.rules., n=10, by="lift")
plot(subRules2, method="paracoord")
