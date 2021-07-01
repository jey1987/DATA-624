library(mlbench)
library(caret)
library(AppliedPredictiveModeling)
library(kernlab)


set.seed(200)
trainingData <- mlbench.friedman1(200, sd = 1)
## We convert the ' x ' data from a matrix to a data frame
## One reason is that this will give the columns names.
trainingData$x <- data.frame(trainingData$x)
## Look at the data using
#featurePlot(trainingData$x, trainingData$y)
## or other methods.
## This creates a list with a vector ' y ' and a matrix
## of predictors ' x ' . Also simulate a large test set to
## estimate the true error rate with good precision:
testData <- mlbench.friedman1(5000, sd = 1)
testData$x <- data.frame(testData$x)


knnModel <- train(x = trainingData$x, y = trainingData$y, method = "knn", preProc = c("center", "scale"), tuneLength = 10)
knnModel

knnPred <- predict(knnModel, newdata = testData$x)
## The function 'postResample' can be used to get the test set  
## performance values
postResample(pred = knnPred, obs = testData$y)


svmModel <- train(x = trainingData$x,
                  y = trainingData$y,
                  method = "svmRadial",
                  tuneLength=10,
                  preProc = c("center", "scale"))
svmModel

svmPred <- predict(svmModel, newdata = testData$x)
postResample(pred = svmPred, obs = testData$y)



nnetGrid <- expand.grid(.decay=c(0, 0.01, 0.1, 0.5, 0.9),
                        .size=c(1, 10, 15, 20),
                        .bag=FALSE)

nnetModel <- train(x = trainingData$x,
                   y = trainingData$y,
                   method = "avNNet",
                   tuneGrid = nnetGrid,
                   preProc = c("center", "scale"),
                   trace=FALSE,
                   linout=TRUE,
                   maxit=500)

nnetModel

nnetPred <- predict(nnetModel, newdata = testData$x)
postResample(pred = nnetPred, obs = testData$y)



marsGrid <- expand.grid(.degree=1:2,
                        .nprune=2:20)

marsModel <- train(x = trainingData$x,
                   y = trainingData$y,
                   method = "earth",
                   tuneGrid = marsGrid,
                   preProc = c("center", "scale"))

marsModel

marsPred <- predict(marsModel, newdata = testData$x)
postResample(pred = marsPred, obs = testData$y)


varImp(marsModel)