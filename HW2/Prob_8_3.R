
library(gbm)
library(AppliedPredictiveModeling)

data(solubility)

grid1 <- expand.grid(n.trees=100, interaction.depth=1, shrinkage=0.1, n.minobsinnode=10)
gbm1 <- train(x = solTrainXtrans, y = solTrainY, method = 'gbm', tuneGrid = grid1, verbose = FALSE)

grid2 <- expand.grid(n.trees=100, interaction.depth=10, shrinkage=0.1, n.minobsinnode=10)
gbm2 <- train(x = solTrainXtrans, y = solTrainY, method = 'gbm', tuneGrid = grid2, verbose = FALSE)
  
varImp(gbm1)
varImp(gbm2)