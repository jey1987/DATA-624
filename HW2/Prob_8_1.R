library(mlbench)
set.seed(200)
simulated <- mlbench.friedman1(200, sd = 1)
simulated <- cbind(simulated$x, simulated$y)
simulated <- as.data.frame(simulated)
colnames(simulated)[ncol(simulated)] <- "y"


library(randomForest)
library(caret)
model1 <- randomForest(y ~ ., 
                       data = simulated,
                       importance = TRUE,
                       ntree = 1000)
rfImp1 <- varImp(model1, scale = FALSE)

rfImp1


simulated$duplicate1 <- simulated$V1 + rnorm(200) * .1
cor(simulated$duplicate1, simulated$V1)


model2 <- randomForest(y ~ ., 
                          data = simulated,
                          importance = TRUE,
                          ntree = 1000)
rfImp2 <- varImp(model2, scale = FALSE)
rfImp2

library(partykit)
library(dplyr)
cforest_model <- cforest(y ~ ., data=simulated)

# Unconditional importance measure
varimp(cforest_model) %>% sort(decreasing = T)

varimp(cforest_model, conditional=T) %>% sort(decreasing = T)


library(gbm)
gbm_Model <- gbm(y ~ ., data=simulated, distribution='gaussian')
summary(gbm_Model)

library(Cubist)
cubistModel <- cubist(x=simulated[,-(ncol(simulated)-1)], y=simulated$y, committees=100)
varImp(cubistModel)



