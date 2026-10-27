getwd()
setwd("D:/Rbag")

# Optimization of machine learning regression model.........................................................................
library(xgboost)
library(rfPermute)
library(ggplot2)
library(A3)
library(reshape2)
library(caret)
library(readr)
library(ranger)    
library(dplyr) 
library(e1071)
library(randomForest)
library(doParallel)   
library(foreach)      

data1 <- read.csv("245 samples top 49 ASVs all - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("245 samples top 49 ASVs all - aquaculture effluent source.csv", header = TRUE, row.names = 1)
data2 <- read.csv("validation set 2 top 49 ASVs all - cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("validation set 2 top 49 ASVs all - aquaculture effluent source.csv", header = TRUE, row.names = 1)

test_data2 <- data2

set.seed(1111)
train_index <- createDataPartition(data1$Group, p = 0.7, list = FALSE)
train_data <- data1[train_index, ]
test_data <- data1[-train_index, ]

if (exists("cl")) try(stopCluster(cl))
try(stopImplicitCluster())
library(doParallel)
registerDoSEQ()

cl <- makePSOCKcluster(4)   
registerDoParallel(cl)

#1.XGBoost..................................................................
x_train <- train_data %>% select(-1)  
y_train <- train_data$Group                 
x_test <- test_data %>% select(-1)
y_test <- test_data$Group
x_valid <- test_data2 %>% select(-1)
y_valid <- test_data2$Group

dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(x_test), label = y_test)
dvalid <- xgb.DMatrix(data = as.matrix(x_valid), label = y_valid)

xgb_grid <- expand.grid(
  nrounds = c(200,300,400),
  max_depth = c(3,5,7),
  eta = c(0.01, 0.1,1),
  gamma = c(0.01, 0.1,1),
  colsample_bytree = c(0.5,0.7,1),
  min_child_weight = c(4,5,6),
  subsample = c(0.1, 0.3,0.5)
)

ctrl <- trainControl(
  method = "cv",
  number = 5,  
  verboseIter = TRUE,
  allowParallel = TRUE
)

set.seed(444)
xgb_tune <- train(
  x = x_train,             
  y = y_train,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = xgb_grid,
  metric = "RMSE"
)

cat("optimal parameters：\n")
print(xgb_tune$bestTune)

final_model <- xgboost(
  data = dtrain,
  nrounds = xgb_tune$bestTune$nrounds,
  max_depth = xgb_tune$bestTune$max_depth,
  eta = xgb_tune$bestTune$eta,
  gamma = xgb_tune$bestTune$gamma,
  colsample_bytree = xgb_tune$bestTune$colsample_bytree,
  min_child_weight = xgb_tune$bestTune$min_child_weight,
  subsample = xgb_tune$bestTune$subsample,
  objective = "reg:squarederror",  
  eval_metric = "rmse",
  early_stopping_rounds = 20,
  verbose = 1
)

pred_train <- predict(final_model, dtrain)
postResample(pred = pred_train, obs = y_train) %>% print()

pred_test <- predict(final_model, dtest)
postResample(pred = pred_test, obs = y_test) %>% print()
pred_valid <- predict(final_model, dvalid)
postResample(pred = pred_valid, obs = y_valid) %>% print()


#2.RF........................................................................
set.seed(111)
ctrl <- trainControl(method = "cv", 
                     number = 5,
                     search = "grid")
tuneGrid <- expand.grid(mtry = 1:10)

set.seed(1111)
otu.train.forest.tuned <- train(Group ~ ., 
                                data = train_data,
                                method = "rf",
                                trControl = ctrl,
                                tuneGrid = tuneGrid,
                                ntree = 2000,
                                importance = TRUE)
print(otu.train.forest.tuned)
best_mtry <- otu.train.forest.tuned$bestTune$mtry

set.seed(1111)
otu.train.forest1 <- randomForest(Group ~ ., 
                                  data = train_data,
                                  mtry = best_mtry,
                                  ntree = 2000,
                                  importance = TRUE)

otu.train.forest1

trainpred<-predict(otu.train.forest1, newdata = train_data)
trainpred
defaultSummary(data.frame(obs=train_data$Group, pred=trainpred))

testpred<-predict(otu.train.forest1, newdata = test_data)
testpred
defaultSummary(data.frame(obs=test_data$Group, pred=testpred))



testpred2<-predict(otu.train.forest1, newdata = test_data2)
testpred2
defaultSummary(data.frame(obs=test_data2$Group, pred=testpred2))

#3.SVM-radial
ctrl <- trainControl(
  method = "cv",        
  number = 5,           
  verboseIter = FALSE,  
  returnResamp = "final", 
  allowParallel = TRUE  
)

svm_radial <- train(Group ~ .,
                    data = train_data,
                    method = "svmRadial",
                    tuneGrid = expand.grid(C = 10^seq(-1, 2, by = 0.1),
                                           sigma = 10^seq(-1, 2, by = 0.1)),
                    trControl = ctrl)
train2<- predict(svm_radial, train_data)
train2
train_performance <- postResample(pred = train2, obs = train_data$Group)
print(train_performance)

predictions2 <- predict(svm_radial,test_data)
predictions2
performance <- postResample(pred = predictions2, obs = test_data$Group)
print(performance)

predictions3 <- predict(svm_radial,test_data2)
predictions3
performance <- postResample(pred = predictions3, obs = test_data2$Group)
print(performance)

#4.SVM-poly
svm_poly <- train(Group ~ .,
                  data = train_data,
                  method = "svmPoly",
                  tuneGrid = expand.grid(C = 10^seq(-2, 2, by = 1),
                                         scale = c(0.01, 0.1, 1),
                                         degree = c(2, 3, 4)),
                  trControl = ctrl)
train3<- predict(svm_poly, train_data)
train3
train_performance <- postResample(pred = train3, obs = train_data$Group)
print(train_performance)

predictions3 <- predict(svm_poly, test_data)
predictions3
performance <- postResample(pred = predictions3, obs = test_data$Group)
print(performance)

predictions4 <- predict(svm_poly, test_data2)
predictions4
performance <- postResample(pred = predictions4, obs = test_data2$Group)
print(performance)

#5.SVM-sigmoid
svm_sigmoid <- svm(Group ~ .,
                   data = train_data,
                   tuneGrid = expand.grid(
                     C = 10^seq(-2, 2, by = 0.1),
                     sigma = 10^seq(-3, 2, by = 0.1),  
                     coef0 = seq(-1, 1, by = 0.1)
                   ),
                   kernel = "sigmoid",
                   trControl = ctrl)  
svm_sigmoid
train4<- predict(svm_sigmoid, train_data)
train4
train_performance <- postResample(pred = train4, obs = train_data$Group)
print(train_performance)

predictions4 <- predict(svm_sigmoid, test_data)
predictions4
performance <- postResample(pred = predictions4, obs = test_data$Group)
print(performance)

predictions5 <- predict(svm_sigmoid, test_data2)
predictions5
performance <- postResample(pred = predictions5, obs = test_data2$Group)
print(performance)

#6.ANN
tuneGrid <- expand.grid(
  size = seq(1:10),   
  decay = c(0.001, 0.01, 0.1)  
)


ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,          
  repeats = 3,         
  verboseIter = TRUE  
)


nn_model <- train(
  Group ~ .,
  data = train_data,        
  method = "nnet",
  linout = TRUE,       
  trace = FALSE,      
  maxit = 1000,       
  tuneGrid = tuneGrid,
  trControl = ctrl
)

print(paste("best size:", nn_model$bestTune$size))
print(paste("best decay:", nn_model$bestTune$decay))

trains <- predict(nn_model, newdata = train_data)
performance <- postResample(pred = trains, obs = train_data$Group)
print(performance)

predictions <- predict(nn_model, newdata = test_data)
predictions

performance <- postResample(pred = predictions, obs = test_data$Group)
print(performance)

predictions2 <- predict(nn_model, newdata = test_data2)

performance <- postResample(pred = predictions2, obs = test_data2$Group)
print(performance)

#7.KNN
ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,          
  repeats = 3,         
  verboseIter = TRUE  
)

tuneGrid <- expand.grid(
  k = seq(1, 20, by = 1)  
)


knn_model <- train(
  Group ~ .,
  data = train_data,  
  method = "knn",
  trControl = ctrl,
  tuneGrid = tuneGrid,
  metric = "RMSE"
)


print(knn_model)
cat("best K:", knn_model$bestTune$k, "\n")

trains <- predict(knn_model, newdata = train_data)
performance <- postResample(pred = trains, obs = train_data$Group)
print(performance)

predictions <- predict(knn_model, newdata = test_data)

performance <- postResample(pred = predictions, obs = test_data$Group)
print(performance)

predictions2 <- predict(knn_model, newdata = test_data2)

performance <- postResample(pred = predictions2, obs = test_data2$Group)
print(performance)








#Boruta-Bootstrap.......................................................................
getwd()
setwd("D:/Rbag")
library(Boruta)
library(randomForest)
library(caret)
library(doParallel)
library(foreach)
library(tidyr)   


data1 <- read.csv("top 100 ASVs all - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("top 100 ASVs all - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("top 100 ASVs count selected - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("top 100 ASVs relative selected - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("top 100 ASVs count selected - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("top 100 ASVs relative selected - aquaculture effluent source.csv", header = TRUE, row.names = 1)

X <- data1[, !names(data1) %in% "Group", drop = FALSE]
y <- data1$Group

full_data <- data.frame(Group = y, X)   
set.seed(5555)
n <- nrow(full_data)
B <- 1000          


cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

combine_results <- function(x, y) {
  list(metrics   = rbind(x$metrics, y$metrics),
       preds     = rbind(x$preds,   y$preds),
       features  = c(x$features, y$features))
}

init <- list(metrics = NULL, preds = NULL, features = list())


final_results <- foreach(i = 1:B,
                         .combine = combine_results,
                         .init = init,
                         .packages = c("randomForest", "caret", "Boruta")) %dopar% {
                           
                           
                           train_idx <- sample(1:n, size = n, replace = TRUE)
                           valid_idx <- setdiff(1:n, unique(train_idx))
                           
                           train_data <- full_data[train_idx, , drop = FALSE]
                           
                           
                           set.seed(i)   
                           boruta_train <- Boruta(
                             x = train_data[, -1, drop = FALSE],
                             y = train_data$Group,
                             doTrace = 0,       
                             maxRuns = 100
                           )
                           selected_features <- getSelectedAttributes(boruta_train, withTentative = FALSE)
                           
                           
                           if (length(selected_features) == 0) {
                             return(list(
                               metrics = data.frame(iter = i, cv_R2 = NA, cv_RMSE = NA,
                                                    train_R2 = NA, train_RMSE = NA,
                                                    valid_R2 = NA, valid_RMSE = NA),
                               preds = NULL,
                               features = list(character(0))
                             ))
                           }
                           
                           train_data_sel <- train_data[, c("Group", selected_features), drop = FALSE]
                           
                           
                           max_mtry <- min(10, length(selected_features))
                           ctrl <- trainControl(
                             method = "cv",
                             number = 5,
                             savePredictions = FALSE,
                             summaryFunction = defaultSummary
                           )
                           tune_grid <- expand.grid(mtry = 1:max_mtry)
                           
                           rf_cv <- tryCatch({
                             train(
                               x = train_data_sel[, selected_features, drop = FALSE],
                               y = train_data_sel$Group,
                               method = "rf",
                               trControl = ctrl,
                               tuneGrid = tune_grid,
                               ntree = 2000,
                               importance = FALSE,
                               verbose = FALSE
                             )
                           }, error = function(e) NULL)
                           
                           if (is.null(rf_cv)) {
                             return(list(
                               metrics = data.frame(iter = i, cv_R2 = NA, cv_RMSE = NA,
                                                    train_R2 = NA, train_RMSE = NA,
                                                    valid_R2 = NA, valid_RMSE = NA),
                               preds = NULL,
                               features = list(selected_features)
                             ))
                           }
                           
                           best_metrics <- rf_cv$results[which.min(rf_cv$results$RMSE), ]
                           cv_RMSE <- best_metrics$RMSE
                           cv_R2   <- best_metrics$Rsquared
                           final_model <- rf_cv$finalModel
                           
                           
                           train_pred <- predict(final_model,
                                                 newdata = train_data_sel[, selected_features, drop = FALSE])
                           train_obs  <- train_data_sel$Group
                           train_metrics <- postResample(train_pred, train_obs)
                           train_rmse <- train_metrics[1]
                           train_r2   <- train_metrics[2]
                           
                           train_results <- data.frame(
                             iter = i,
                             dataset = "train",
                             sample_id = rownames(train_data_sel),
                             observed = train_obs,
                             predicted = train_pred,
                             stringsAsFactors = FALSE
                           )
                           
                           
                           if (length(valid_idx) == 0) {
                             valid_rmse <- NA
                             valid_r2   <- NA
                             valid_results <- NULL
                           } else {
                             valid_data <- full_data[valid_idx, , drop = FALSE]
                             valid_data_sel <- valid_data[, c("Group", selected_features), drop = FALSE]
                             valid_pred <- predict(final_model,
                                                   newdata = valid_data_sel[, selected_features, drop = FALSE])
                             valid_obs  <- valid_data_sel$Group
                             valid_metrics <- postResample(valid_pred, valid_obs)
                             valid_rmse <- valid_metrics[1]
                             valid_r2   <- valid_metrics[2]
                             
                             valid_results <- data.frame(
                               iter = i,
                               dataset = "valid",
                               sample_id = rownames(valid_data_sel),
                               observed = valid_obs,
                               predicted = valid_pred,
                               stringsAsFactors = FALSE
                             )
                           }
                           
                           
                           preds_this <- do.call(rbind, list(train_results, valid_results))
                           
                          
                           metrics_this <- data.frame(
                             iter = i,
                             cv_R2 = cv_R2,
                             cv_RMSE = cv_RMSE,
                             train_R2 = train_r2,
                             train_RMSE = train_rmse,
                             valid_R2 = valid_r2,
                             valid_RMSE = valid_rmse
                           )
                           
                           list(metrics = metrics_this,
                                preds = preds_this,
                                features = list(selected_features))
                         }

stopCluster(cl)

metrics_all <- final_results$metrics
predictions_all <- final_results$preds
features_list <- final_results$features

valid_iter <- !is.na(metrics_all$cv_R2)
metrics_all <- metrics_all[valid_iter, ]
predictions_all <- predictions_all[predictions_all$iter %in% metrics_all$iter, ]
features_list <- features_list[valid_iter]

all_selected <- unlist(features_list)
freq_table <- table(all_selected)
freq <- freq_table / length(features_list)   # 基于有效迭代数的频率
stable_features <- names(freq[freq >= 0.6])











#RF regression model construct
getwd()
setwd("D:/Rbag")

library(glmnet)
library(Boruta)
library(randomForest)
library(caret)       
library(doParallel)
library(foreach)
library(tidyr)

data1 <- read.csv("245 samples Boruta0.6 relative all - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("245 samples Boruta0.7 relative all - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("245 samples Boruta0.8 relative all - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("245 samples Boruta0.9 relative all - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("245 samples Boruta1 relative all - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("101 samples Boruta0.6 relative selected - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("101 samples Boruta0.7 relative selected - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("101 samples Boruta0.8 relative selected - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("101 samples Boruta0.9 relative selected - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("101 samples Boruta1 relative selected - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("101 samples Boruta0.6 count selected - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("101 samples Boruta0.7 count selected - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("101 samples Boruta0.8 count selected - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("101 samples Boruta0.9 count selected - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("101 samples Boruta1 count selected - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("101 samples Boruta1 relative selected with dual probability - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("101 samples Boruta1 relative selected with max probability - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("101 samples Boruta1 relative selected with min probability - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("245 samples Boruta0.6 relative all - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("245 samples Boruta0.7 relative all - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("245 samples Boruta0.8 relative all - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("245 samples Boruta0.9 relative all - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("245 samples Boruta1 relative all - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("154 samples Boruta0.6 relative selected - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("154 samples Boruta0.7 relative selected - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("154 samples Boruta0.8 relative selected - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("154 samples Boruta0.9 relative selected - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("154 samples Boruta1 relative selected - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("154 samples Boruta0.6 count selected - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("154 samples Boruta0.7 count selected - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("154 samples Boruta0.8 count selected - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("154 samples Boruta0.9 count selected - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("154 samples Boruta1 count selected - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("154 samples Boruta0.9 relative selected with dual probability - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("154 samples Boruta0.9 relative selected with max probability - cropland runoff source.csv", header = TRUE, row.names = 1)
#data1 <- read.csv("154 samples Boruta0.9 relative selected with min probability - cropland runoff source.csv", header = TRUE, row.names = 1)

data2 <- read.csv("100 samples validation set 2 Boruta0.6 relative all -aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("100 samples validation set 2 Boruta0.7 relative all -aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("100 samples validation set 2 Boruta0.8 relative all -aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("100 samples validation set 2 Boruta0.9 relative all -aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("100 samples validation set 2 Boruta1 relative all -aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("39 samples validation set 2 Boruta0.6 relative selected -aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("39 samples validation set 2 Boruta0.7 relative selected -aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("39 samples validation set 2 Boruta0.8 relative selected -aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("39 samples validation set 2 Boruta0.9 relative selected -aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("39 samples validation set 2 Boruta1 relative selected -aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("39 samples validation set 2 Boruta0.6 count selected -aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("39 samples validation set 2 Boruta0.7 count selected -aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("39 samples validation set 2 Boruta0.8 count selected -aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("39 samples validation set 2 Boruta0.9 count selected -aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("39 samples validation set 2 Boruta1 count selected -aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("39 samples Boruta1 relative selected with dual probability - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("39 samples Boruta1 relative selected with max probability - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("39 samples Boruta1 relative selected with min probability - aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("100 samples validation set 2 Boruta0.6 relative all -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("100 samples validation set 2 Boruta0.7 relative all -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("100 samples validation set 2 Boruta0.8 relative all -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("100 samples validation set 2 Boruta0.9 relative all -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("100 samples validation set 2 Boruta1 relative all -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("69 samples validation set 2 Boruta0.6 relative selected -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("69 samples validation set 2 Boruta0.7 relative selected -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("69 samples validation set 2 Boruta0.8 relative selected -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("69 samples validation set 2 Boruta0.9 relative selected -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("69 samples validation set 2 Boruta0.1 relative selected -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("69 samples validation set 2 Boruta0.6 count selected -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("69 samples validation set 2 Boruta0.7 count selected -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("69 samples validation set 2 Boruta0.8 count selected -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("69 samples validation set 2 Boruta0.9 count selected -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("69 samples validation set 2 Boruta1 count selected -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("69 samples validation set 2 Boruta0.9 relative selected with dual probability -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("69 samples validation set 2 Boruta0.9 relative selected with max probability -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("69 samples validation set 2 Boruta0.9 relative selected with min probability -cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("positive sink samples wet season Boruta0.9 relative selected with min probability- cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("positive sink samples dry season Boruta0.9 relative selected with min probability- cropland runoff source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("positive sink samples dry season Boruta1 relative selected with dual probability- aquaculture effluent source.csv", header = TRUE, row.names = 1)
#data2 <- read.csv("positive sink samples wet season Boruta1 relative selected with dual probability- aquaculture effluent source.csv", header = TRUE, row.names = 1)


X1 <- data1[, !names(data1) %in% "Group", drop = FALSE]
y1 <- data1$Group
X2 <- data2[, !names(data2) %in% "Group", drop = FALSE]
y2 <- data2$Group

set.seed(1111)

data_full <- cbind(X1, Group = y1)
n <- nrow(data_full)
B <- 1000            


test2_data <- data2


cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)


combine_results <- function(x, y) {
  list(metrics = rbind(x$metrics, y$metrics),
       preds   = rbind(x$preds,   y$preds))
}

init <- list(metrics = NULL, preds = NULL)


final_results <- foreach(i = 1:B,
                         .combine = combine_results,
                         .init = init,
                         .packages = c("randomForest", "caret")) %dopar% {
                           
                           
                           train_idx <- sample(1:n, size = n, replace = TRUE)
                           valid_idx <- setdiff(1:n, unique(train_idx))
                           
                           train_data <- data_full[train_idx, , drop = FALSE]
                           
                          
                           ctrl <- trainControl(
                             method = "cv",
                             number = 5,
                             savePredictions = FALSE,
                             summaryFunction = defaultSummary
                           )
                           tune_grid <- expand.grid(mtry = 1:10)
                           
                           rf_cv <- train(
                             x = train_data[, !names(train_data) %in% "Group", drop = FALSE],
                             y = train_data$Group,
                             method = "rf",
                             trControl = ctrl,
                             tuneGrid = tune_grid,
                             ntree = 2000,         
                             importance = FALSE,
                             verbose = FALSE
                           )
                           
                           best_metrics <- rf_cv$results[which.min(rf_cv$results$RMSE), ]
                           cv_RMSE <- best_metrics$RMSE
                           cv_R2   <- best_metrics$Rsquared
                           
                           final_model <- rf_cv$finalModel
                           
                           
                           train_pred <- predict(final_model,
                                                 newdata = train_data[, !names(train_data) %in% "Group", drop = FALSE])
                           train_obs  <- train_data$Group
                           train_metrics <- postResample(train_pred, train_obs)
                           train_rmse <- train_metrics[1]
                           train_r2   <- train_metrics[2]
                           
                           train_results <- data.frame(
                             iter = i,
                             dataset = "train",
                             sample_id = rownames(train_data),
                             observed = train_obs,
                             predicted = train_pred,
                             stringsAsFactors = FALSE
                           )
                           
                           
                           if (length(valid_idx) == 0) {
                             valid_rmse <- NA
                             valid_r2   <- NA
                             valid_results <- NULL
                           } else {
                             valid_data <- data_full[valid_idx, , drop = FALSE]
                             valid_pred <- predict(final_model,
                                                   newdata = valid_data[, !names(valid_data) %in% "Group", drop = FALSE])
                             valid_obs  <- valid_data$Group
                             valid_metrics <- postResample(valid_pred, valid_obs)
                             valid_rmse <- valid_metrics[1]
                             valid_r2   <- valid_metrics[2]
                             
                             valid_results <- data.frame(
                               iter = i,
                               dataset = "valid",
                               sample_id = rownames(valid_data),
                               observed = valid_obs,
                               predicted = valid_pred,
                               stringsAsFactors = FALSE
                             )
                           }
                           
                           
                           test2_pred <- predict(final_model,
                                                 newdata = test2_data[, !names(test2_data) %in% "Group", drop = FALSE])
                           test2_obs  <- test2_data$Group
                           test2_metrics <- postResample(test2_pred, test2_obs)
                           test2_rmse <- test2_metrics[1]
                           test2_r2   <- test2_metrics[2]
                           
                           test2_results <- data.frame(
                             iter = i,
                             dataset = "test2",
                             sample_id = rownames(test2_data),
                             observed = test2_obs,
                             predicted = test2_pred,
                             stringsAsFactors = FALSE
                           )
                           
                           
                           preds_this <- do.call(rbind, list(train_results, valid_results, test2_results))
                           
                           
                           metrics_this <- data.frame(
                             iter = i,
                             cv_R2 = cv_R2,
                             cv_RMSE = cv_RMSE,
                             train_R2 = train_r2,
                             train_RMSE = train_rmse,
                             valid_R2 = valid_r2,
                             valid_RMSE = valid_rmse,
                             test2_R2 = test2_r2,
                             test2_RMSE = test2_rmse
                           )
                           
                           list(metrics = metrics_this, preds = preds_this)
                         }

stopCluster(cl)


metrics_all <- final_results$metrics
predictions_all <- final_results$preds



for (ds in c("train", "valid", "test2")) {
  
  sub <- subset(predictions_all, dataset == ds)
  
  if (nrow(sub) == 0) {
    warning(paste("dataset", ds, "pass"))
    next
  }
  
  
  wide <- pivot_wider(sub,
                      id_cols = sample_id,
                      names_from = iter,
                      values_from = predicted)
  
  
  out <- cbind(sample_id = wide$sample_id, as.data.frame(wide[, -1, drop = FALSE]))
  rownames(out) <- NULL
  
 
  filename <- paste0("bootstrap_preds_", ds, ".csv")
  write.csv(out, filename, row.names = FALSE)
  
  cat(paste("prediction matrix saved:", filename, "\n"))
}


all_preds <- predictions_all
all_preds$unique_id <- paste0(all_preds$dataset, "_", all_preds$sample_id)
all_wide <- pivot_wider(all_preds, id_cols = unique_id, names_from = iter, values_from = predicted)