install.packages("rsample")
install.packages("dplyr")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("ipred")
install.packages("caret")
install.packages("lime")
install.packages("h2o")
install.packages("ranger")
install.packages("pdp")
install.packages("xgboost")
install.packages("vtreat")

library(rsample)     # data splitting 
library(dplyr)       # data wrangling
library(rpart)       # performing regression trees
library(rpart.plot)  # plotting regression trees
library(ipred)       # bagging
library(caret)       # bagging
library(tree)        # decision tree
library(randomForest)# random forest
library(gbm)         # gradient boosting
library(ranger)       # a faster implementation of randomForest
library(h2o)          # an extremely fast java-based platform
library(pdp)          # model visualization
library(ggplot2)      # model visualization
library(lime)         # model visualization
library(gbm)          # basic implementation
library(xgboost)      # a faster implementation of gbm
library(vtreat)
library(xlsx)
library(cluster) 
#-------------------------------------------------------------------------
# Setting up the training and testing dataset
set.seed(123)
setwd("C:\\Users\\User\\Documents")
data = read.csv('vipdatamodelling6.csv', header = TRUE, sep=",")
data[,1] = NULL
data[,8] = NULL
colnames(data)[1] <- c("market_price")
data_split <- initial_split(data, prop = .7)
data_train <- training(data_split)
data_test  <- testing(data_split)
#-------------------------------------------------------------------------


#-------------------------------------------------------------------------
# Model 1: Regression trees by CART algorithm
m1 <- rpart(
  formula = target2 ~ .,
  data    = data_train,
  method  = "anova" # for regression trees
)
m1
rpart.plot(m1) # single CART regression tree plot
plotcp(m1) # y-axis is cv error for Cost Complexity , x-axis is cost complexity alpha tuning parameter 
# plotcp helps us to select the number of terminal nodes
# it is shown that selecting a tree with 11 terminal nodes has the lowest CV error for cost complexity criterion

# Parameter pruning on regression tree on alpha
m2 <- rpart( # based on the plotcp we insert control of cp=0
  formula = target2 ~ .,
  data    = data_train,
  method  = "anova", 
  control = list(cp = 0, xval = 10) # by inserting control we can prune the tree
) # x-val = 10, means 10 fold cross validation
rpart.plot(m2)
plotcp(m2)
abline(v = 11, lty = "dashed") # with 7 terminal nodes and 6 splits
m1$cptable # Create Conditional Probability Tables (CPTs)
m2$cptable

# now we perform parameter tuning of minsplit and maxdepth
# minsplit: the minimum number of data points required to attempt a split before it is forced to create a terminal node. 
# The default is 20. Making this smaller allows for terminal nodes that may contain only a handful of observations to create the predicted value.
# maxdepth: the maximum number of internal nodes between the root node and the terminal nodes. 
# The default is 30, which is quite liberal and allows for fairly large trees to be built.

# gridsearch
hyper_grid <- expand.grid(
  minsplit = seq(5, 30, 1),
  maxdepth = seq(5, 30, 1)
)
models <- list()

for (i in 1:nrow(hyper_grid)) {
  
  # get minsplit, maxdepth values at row i
  minsplit <- hyper_grid$minsplit[i] # minsplit:minimum number of observations that must exist in a node for a split to be attempted.
  maxdepth <- hyper_grid$maxdepth[i] # maxdepth: Set the maximum depth of any node of the final tree
  
  # train a model and store in the list
  models[[i]] <- rpart(
    formula = target2 ~ .,
    data    = data_train,
    method  = "anova",
    control = list(minsplit = minsplit, maxdepth = maxdepth)
  )
}
# function to get optimal cp
get_cp <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  cp <- x$cptable[min, "CP"] 
}
# function to get minimum error
get_min_error <- function(x) {
  min    <- which.min(x$cptable[, "xerror"])
  xerror <- x$cptable[min, "xerror"] 
}
hyper_grid %>%
  mutate(
    cp    = purrr::map_dbl(models, get_cp),
    error = purrr::map_dbl(models, get_min_error)
  ) %>%
  arrange(error) %>%
  top_n(-5, wt = error) # top 5 permutations of params
optimal_tree <- rpart(
  formula = target2 ~ .,
  data    = data_train,
  method  = "anova",
  control = list(minsplit = 9, maxdepth = 20, cp = 0.01)
)
optimal_tree$cptable # psuedo rmse = xerror 
pred = predict(optimal_tree, newdata = data_test)
obs = data_test$target2
# Function that returns Root Mean Squared Error
mse <- function(pred,obs)
{
  mean((pred-obs)^2)
}

# Function that returns Mean Absolute Error
mae <- function(pred,obs)
{
  mean(abs(pred-obs))
}
# Results for Model 1: Regression trees by CART algorithm
mse(pred,obs)
mae(pred,obs)
pred1 = predict(m1, newdata = data_test)
mse(pred1,obs) # Results on Table 7.2
mae(pred1,obs) #  Results on Table 7.2
rpart.plot(optimal_tree)  # Figure 7.1
# ------------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Model 2: Regression trees (Bagging) 

set.seed(123)

# train bagged model

ntree <- 10:50

# create empty vector to store OOB RMSE values
rmse <- vector(mode = "numeric", length = length(ntree))

for (i in seq_along(ntree)) {
  # reproducibility
  set.seed(123)
  
  # perform bagged model
  model <- bagging(
    formula = target2 ~ .,
    data    = data_train,
    coob    = TRUE, #coob = TRUE to use the OOB sample to estimate the pseudo test error
    nbagg   = ntree[i]
  )
  # get OOB error
  rmse[i] <- model$err
}

plot(ntree, rmse, type = 'l', lwd = 2) # Figure 7.2
abline(v =29, col = "red", lty = "dashed")

ctrl <- trainControl(method = "cv",  number = 10) 

# CV bagged model
bagged_cv <- train(
  target2 ~ .,
  data = data_train,
  method = "treebag",
  trControl = ctrl,
  importance = TRUE,
  nbagg = 29
)

# assess results
bagged_cv


# plot most important variables
plot(varImp(bagged_cv), 6)  # Figure 7.3

# Results for Model 2: Regression trees (bagging)
pred <- predict(bagged_cv, data_test)
obs = data_test$target2
mse(pred,obs) # Results on Table 7.3
mae(pred,obs) # Results on Table 7.2
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# random forest (model not used)
# for reproduciblity
# set.seed(123)
# 
# # default RF model
# # This gives us an indication of how many trees to select for tuning
# #plotted error rate above is based on the OOB sample error and can be accessed directly at m1$mse
# m1 <- randomForest(
#   formula = target2 ~ .,
#   data    = data_train
# )
# m1
# plot(m1)
# # number of trees with lowest MSE
# which.min(m1$mse) # number of trees = 15
# # MSE of this optimal random forest
# m1$mse[which.min(m1$mse)]
# 
# # create pseudo train and test data for psuedo test error 
# # we want to do cross comparison of OOB errors and pseudo test error by MSE
# set.seed(123)
# valid_split <- initial_split(data_train, .8) # 80% of training data used as pseudo train data
# pseudo_train <- analysis(valid_split)
# pseudo_test <- assessment(valid_split)
# x_test <- pseudo_test[setdiff(names(data), "target2")]
# y_test <- pseudo_test$target2
# 
# # comparison between OOB errors and pseudo Test errors based on splitting of training sets
# rf_oob_comp <- randomForest(
#   formula = target2 ~ .,
#   data    = pseudo_train,
#   xtest   = x_test,
#   ytest   = y_test
# )
# 
# # extract OOB & validation errors
# oob <- (rf_oob_comp$mse)
# validation <- (rf_oob_comp$test$mse)
# tibble::tibble(
#   `Out of Bag Error` = oob,
#   `Test error` = validation,
#   ntrees = 1:rf_oob_comp$ntree
# ) %>%
#   gather(Metric, MSE, -ntrees) %>%
#   ggplot(aes(ntrees, MSE, color = Metric)) +
#   geom_line() +
#   scale_y_continuous(labels = scales::dollar) +
#   xlab("Number of trees")
# 
# # tuning of parameters
# # ntree: number of trees , mtry: the number of variables to randomly sample, mtry =p , the model equates to bagging
# # sampsize: the number of samples to train on
# # nodesize: minimum number of samples within the terminal nodes
# # maxnodes: maximum number of terminal nodes
# 
# features <- setdiff(names(data), "target2")
# set.seed(123)
# 
# m2 <- tuneRF(
#   x          = data_train[features],
#   y          = data_train$target2,
#   ntreeTry   = 500,
#   mtryStart  = 1,
#   stepFactor = 1,
#   improve    = 0.01,
#   trace      = FALSE      # to not show real-time progress 
# )
# 
# # hyperparameter grid search
# hyper_grid <- expand.grid(
#   mtry       = seq(1, 5, by = 1),
#   node_size  = seq(3, 10, by = 1),
#   sampe_size = c(.55, .632, .70, .80),
#   OOB_MSE   = 0 # psuedo test error MSE
# )
# 
# for(i in 1:nrow(hyper_grid)) {
#   
#   # train model
#   model <- ranger(
#     formula         = target2 ~ ., 
#     data            = data_train, 
#     num.trees       = 500,
#     mtry            = hyper_grid$mtry[i],
#     min.node.size   = hyper_grid$node_size[i],
#     sample.fraction = hyper_grid$sampe_size[i],
#     seed            = 123
#   )
#   
#   # add OOB error to grid
#   hyper_grid$OOB_MSE[i] <- (model$prediction.error)
# }
# 
# hyper_grid %>% 
#   dplyr::arrange(OOB_MSE) %>%
#   head(10)
# 
# OOB_RMSE <- vector(mode = "numeric", length = 100)
# 
# for(i in seq_along(OOB_RMSE)) {
#   
#   optimal_ranger <- ranger(
#     formula         = target2 ~ ., 
#     data            = data_train, 
#     num.trees       = 500,
#     mtry            = 24,
#     min.node.size   = 5,
#     sample.fraction = .8,
#     importance      = 'impurity'
#   )
#   
#   OOB_RMSE[i] <- sqrt(optimal_ranger$prediction.error)
# }
# 
# hist(OOB_RMSE, breaks = 20)
# optimal_ranger$variable.importance %>% 
#   tidy() %>%
#   dplyr::arrange(desc(x)) %>%
#   dplyr::top_n(25) %>%
#   ggplot(aes(reorder(names, x), x)) +
#   geom_col() +
#   coord_flip() +
#   ggtitle("Top 5 important variables")
# 
# # ranger
# pred_ranger <- predict(optimal_ranger, data_test)
# obs = data_test$target2
# mse(pred_ranger,obs)
# mae(pred_ranger,obs)
# ----------------------------------------------------------

# ----------------------------------------------------------------
# Model 3: Boosted Regression Trees (with GBM)

# Basic implementation of GBM in R package

set.seed(123)

# train GBM model
gbm.fit <- gbm(
  formula = target2 ~ .,
  distribution = "gaussian",
  data = data_train,
  n.trees = 10000,
  interaction.depth = 1, # interaction.depth parameter as a number of splits it has to perform on a tree 
  shrinkage = 0.001,
  cv.folds = 10,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  
print(gbm.fit)
min(gbm.fit$cv.error) # get MSE 
# plot loss function as a result of n trees added to the ensemble
gbm.perf(gbm.fit, method = "cv") # x-axis is no of trees

# tuning of parameters with grid search
# create hyperparameter grid
hyper_grid <- expand.grid(
  shrinkage = c(.01, .1, .3),
  interaction.depth = c(1, 3, 5),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_MSE = 0,                     # a place to dump results
  min_MAE = 0
  )

# to lower computation costs, we use valuation set rather than cross validation
# randomize data
random_index <- sample(1:nrow(data_train), nrow(data_train))
random_data_train <- data_train[random_index, ]

# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune <- gbm(
    formula = target2 ~ .,
    distribution = "gaussian",
    data = random_data_train,
    n.trees = 5000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75, # 75% of data as pseudo training dataset in valuation set approach
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_MSE[i] <- (min(gbm.tune$valid.error))
}

hyper_grid %>% 
  dplyr::arrange(min_MSE) %>%
  head(10)

#These results help us to zoom into areas where we can refine our search.

# we carry out another grid search with a better range of values for each parameters
hyper_grid <- expand.grid(
  shrinkage = c(.01, .05, .1, .3, 0.4, 0.5),
  interaction.depth = c(3, 5, 7),
  n.minobsinnode = c(5, 7, 10),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid)

# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune <- gbm(
    formula = target2 ~ .,
    distribution = "gaussian",
    data = random_data_train,
    n.trees = 6000,
    interaction.depth = hyper_grid$interaction.depth[i],
    shrinkage = hyper_grid$shrinkage[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i],
    train.fraction = .75,
    n.cores = NULL, # will use all cores by default
    verbose = FALSE
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(gbm.tune$valid.error)
  hyper_grid$min_MSE[i] <- min(gbm.tune$valid.error)
}

hyper_grid %>% 
  dplyr::arrange(min_MSE) %>%
  head(10)

# for reproducibility
set.seed(123)

# train GBM model
gbm.fit.final <- gbm(
  formula = target2 ~ .,
  distribution = "gaussian",
  data = data_train,
  n.trees = 5824,
  interaction.depth = 3,
  shrinkage = 0.05,
  n.minobsinnode = 5,
  bag.fraction = 0.65, 
  train.fraction = 1,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  
par(mar = c(5, 8, 1, 1))
summary(
  gbm.fit.final, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
) # figure 7.4
# understand how the response variable changes based on these variables.
# PDPs plot the change in the average predicted value as specified feature(s) vary over their marginal distribution.
# e.g. average change in target2 as the sales price changes while holding all other variables constant
gbm.fit.final %>%
  partial(pred.var = "cut_fav_amt", n.trees = gbm.fit.final$n.trees, grid.resolution = 100) %>%
  autoplot(rug = TRUE, train = data_train) # Figure 7.5
# predict values for test data
pred <- predict(gbm.fit.final, n.trees = gbm.fit.final$n.trees, data_test)
#pred <- predict(gbm.fit.final, n.trees = gbm.fit.final$n.trees, data)

# Results for Model 3: Boosted Regression trees (with GBM)
caret::RMSE(pred, data_test$target2) # Results on Table 7.4
caret::MAE(pred, data_test$target2)  # Results on Table 7.4
# ----------------------------------------------------------

# ----------------------------------------------------------
# Model 4: Boosted Regression trees (with XGBoost)

set.seed(123)
# setting the initial default values for xgboost
# learning rate (eta): 0.3
# tree depth (max_depth): 6
# minimum node size (min_child_weight): 1
# percent of training data to sample for each tree (subsample -> equivalent to gbm's bag.fraction): 100%
features <- setdiff(names(data_train), "target2")

# Create the treatment plan from the training data
treatplan <- vtreat::designTreatmentsZ(data_train, features, verbose = FALSE)

# Get the "clean" variable names from the scoreFrame
new_vars <- treatplan %>%
  magrittr::use_series(scoreFrame) %>%        
  dplyr::filter(code %in% c("clean", "lev")) %>% 
  magrittr::use_series(varName)     

# Prepare the training data
features_train <- vtreat::prepare(treatplan, data_train, varRestriction = new_vars) %>% as.matrix()
response_train <- data_train$target2

# Prepare the test data
features_test <- vtreat::prepare(treatplan, data_test, varRestriction = new_vars) %>% as.matrix()
response_test <- data_test$target2


xgb.fit1 <- xgb.cv(
  data = features_train,
  label = response_train,
  nrounds = 1000,
  nfold = 10, # 10-fold cv
  objective = "reg:linear",  # for regression models
  verbose = 0,               # silent,
  early_stopping_rounds = 10 
)
# xgb.fit1$evaluation_log to identify the minimum RMSE and the optimal number of trees
# get number of trees that minimize error
xgb.fit1$evaluation_log %>%
  dplyr::summarise(
    ntrees.train = which(train_rmse_mean == min(train_rmse_mean))[1],
    rmse.train   = min(train_rmse_mean),
    ntrees.test  = which(test_rmse_mean == min(test_rmse_mean))[1],
    rmse.test   = min(test_rmse_mean), # number of trees required is only 36
  )

# plot error vs number trees
ggplot(xgb.fit1$evaluation_log) +
  geom_line(aes(iter, train_rmse_mean), color = "red") +
  geom_line(aes(iter, test_rmse_mean), color = "blue")

# tuning of parameters for XGboost
# eta (learning rate), max_depth, min_child_weight, subsample, colsample_bytrees
# create hyperparameter grid
hyper_grid <- expand.grid(
  eta = c(.01, .05, .1, .3),
  max_depth = c(1, 3, 5, 7),
  min_child_weight = c(1, 3, 5, 7),
  subsample = c(.65, .8, 1), 
  colsample_bytree = c(.8, .9, 1),
  optimal_trees = 0,               # a place to dump results
  min_MSE = 0,                     # a place to dump results
  min_MAE = 0
  )
nrow(hyper_grid) # no. of models
# grid search 
for(i in 1:nrow(hyper_grid)) {
  
  # create parameter list
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i]
  )

  # reproducibility
  set.seed(123)
  
  # train model
  xgb.tune <- xgb.cv(
    params = params,
    data = features_train,
    label = response_train,
    nrounds = 5000,
    nfold = 5,
    objective = "reg:linear",  # for regression models
    verbose = 0,               # silent,
    early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
  )
  
  # add min training error and trees to grid
  hyper_grid$optimal_trees[i] <- which.min(xgb.tune$evaluation_log$test_rmse_mean)
  hyper_grid$min_RMSE[i] <- min(xgb.tune$evaluation_log$test_rmse_mean)
}

hyper_grid %>%
  dplyr::arrange(min_RMSE) %>%
  head(10)
# optimal parameters
params <- list(
  eta = 0.30,
  max_depth = 5,
  min_child_weight = 1,
  subsample = 0.80,
  colsample_bytree = 0.80
)

# train final model
xgb.fit.final <- xgboost(
  params = params,
  data = features_train,
  label = response_train,
  nrounds = 95,
  objective = "reg:linear",
  verbose = 0
)
# create importance matrix
importance_matrix <- xgb.importance(model = xgb.fit.final)

# variable importance plot
xgb.plot.importance(importance_matrix, top_n = 5, measure = "Gain") # Figure 7.6

pdp <- xgb.fit.final %>%
  partial(pred.var = "cut_fav_amt_clean", n.trees = 95, grid.resolution = 100, train = features_train) %>%
  autoplot(rug = TRUE, train = features_train) +
  scale_y_continuous(labels = scales::dollar) +
  ggtitle("PDP")

# Results for Model 4: Boosted Regression trees (with XGBoost)
# predict values for test data
pred <- predict(xgb.fit.final, n.trees = 95, features_test) # Table 7.5
hist(as.matrix(pred),main="Full-cut promotion demand per order_id",xlab="Full-cut promotion demand",ylab="Frequency")


# results
caret::RMSE(pred, response_test)
caret::MAE(pred, response_test)

features_test <- vtreat::prepare(treatplan, data_test, varRestriction = new_vars) %>% as.matrix()
response_test <- data_test$target2
hist(response_test,main="1Full-cut promotion demand per order_id",xlab="Full-cut promotion demand",ylab="Frequency")
# ----------------------------------------------------------


# ----------------------------------------------------------
# Model 5: Boosted Regression Tree (with h2o)
h2o.no_progress()
h2o.init(max_mem_size = "5g")
##  Connection successful!
# h2o.gbm has the default values for the following parameters:
# number of trees (ntrees): 50
# learning rate (learn_rate): 0.1
# tree depth (max_depth): 5
# minimum observations in a terminal node (min_rows): 10
# no sampling of observations or columns

# turn training set into h2o object
y <- "target2"
x <- setdiff(names(data_train), y)
train.h2o <- as.h2o(data_train)
test.h2o <- as.h2o(data_test)
# training basic GBM model with defaults
h2o.fit1 <- h2o.gbm(
  x = x,
  y = y,
  training_frame = train.h2o,
  nfolds = 10,
  ntrees = 5000,
  stopping_rounds = 10,
  stopping_tolerance = 0,
  seed = 123
)
h2o.fit1
split <- h2o.splitFrame(train.h2o, ratios = 0.75)
train <- split[[1]]
valid <- split[[2]]
# random grid search criteria
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "mse",
  stopping_tolerance = 0.005,
  stopping_rounds = 10,
  max_runtime_secs = 60*60
)

hyper_grid <- list(
  max_depth = c(1, 3, 5),
  min_rows = c(1, 5, 10),
  learn_rate = c(0.01, 0.05, 0.1, 0.2,0.3),
  learn_rate_annealing = c(.99, 1),
  sample_rate = c(.5, .75, 1),
  col_sample_rate = c(.8, .9, 1)
)

# perform grid search 
grid <- h2o.grid(
  algorithm = "gbm",
  grid_id = "gbm_grid2",
  x = x, 
  y = y, 
  training_frame = train,
  validation_frame = valid,
  hyper_params = hyper_grid,
  search_criteria = search_criteria, # add search criteria
  ntrees = 5000,
  stopping_rounds = 10,
  stopping_tolerance = 0,
  seed = 123
)

# collect the results and sort by our model performance metric of choice
grid_perf <- h2o.getGrid(
  grid_id = "gbm_grid2", 
  sort_by = "mse", 
  decreasing = FALSE
)
grid_perf

best_model_id <- grid_perf@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)

# Now let's get performance metrics on the best model
h2o.performance(model = best_model, valid = TRUE)

h2o.final <- h2o.gbm(
  x = x,
  y = y,
  training_frame = train.h2o,
  nfolds = 5,
  ntrees = 10000,
  learn_rate = 0.01,
  learn_rate_annealing = 1,
  max_depth = 3,
  min_rows = 10,
  sample_rate = 0.75,
  col_sample_rate = 1,
  stopping_rounds = 10,
  stopping_tolerance = 0,
  seed = 123
)

# model stopped after xx trees
h2o.final@parameters$ntrees

# cross validated RMSE
h2o.rmse(h2o.final, xval = TRUE)

h2o.varimp_plot(h2o.final, num_of_features = 10) # Figure 7.7
pfun <- function(object, newdata) {
  as.data.frame(predict(object, newdata = as.h2o(newdata)))[[1L]]
}

pdp <- h2o.final %>%
  partial(
    pred.var = "cut_fav_amt", 
    pred.fun = pfun,
    grid.resolution = 20, 
    train = data_train
  ) %>%
  autoplot(rug = TRUE, train = data_train, alpha = .1) +
  scale_y_continuous(labels = scales::dollar) +
  ggtitle("PDP")

# Results for Model 5: Boosted Regression trees (with h2o)
# evaluate performance on new data
h2o.performance(model = h2o.final, newdata = test.h2o) # Results on Table 7.6
# ----------------------------------------------------------
# Model 6: Boosted Regression Tree (with Light GBM) (provided in the python code)
# Model 7: KNN (provided in the python code)
# Model 8: ANN (provided in the python code)

# ----------------------------------------------------------
# ----------------------------------------------------------
# Section 8: Comparison with discrete choice model

# Multinomial logit model
install.packages("foreign")
install.packages("nnet")
install.packages("ggplot2")
install.packages("reshape2")
install.packages("mlogit")
library(foreign)
library(nnet)
library(ggplot2)
library(reshape2)
library(rsample)     # data splitting 
library(dplyr)       # data wrangling
library(mlogit)

# set.seed(123)
# setwd("C:\\Users\\User\\Documents")
# data = read.csv('vipdatamodelling7.csv', header = TRUE)
# colnames(data) <- c("threshold","minprice","priceminprice","target2")
# x <- as.factor(data$target2)
# data$target2 <- relevel(x, ref = "No")
# test <- multinom(target2 ~ threshold + minprice + priceminprice, data = data)
# summary(test)
# z <- summary(test)$coefficients/summary(test)$standard.errors
# z
# p <- (1 - pnorm(abs(z), 0, 1)) * 2
# p
# fit_values <- fitted(test)
# target2<-factor(as.numeric(data$target2))
# levels(target2)[1] <- 0
# levels(target2)[2] <- 1
# count <- 0
# for (i in 1:nrow(fit_values)){
#   if (fit_values[i]==target2[i]){count <- count + 1}
# }
# error_rate <- 1 - count/nrow(fit_values)

data1 = read.csv('vipdatamodelling8.csv', header = TRUE)
colnames(data1) <- c("minprice","priceminprice","target2")
x <- as.factor(data1$target2)
data1_split <- initial_split(data1, prop = .7)
data1_train <- training(data1_split)
data1_test  <- testing(data1_split)
data1$target2 <- relevel(x, ref = "No")
test <- multinom(target2 ~ minprice + priceminprice, data = data1_train)
summary(test)
z <- summary(test)$coefficients/summary(test)$standard.errors
z
p <- (1 - pnorm(abs(z), 0, 1)) * 2
p
#fit_values <- fitted(test, data = data1_test)
fit_values <- predict(test, newdata = data1_test, "probs")
fit_values <- as.matrix(fit_values)
target2<-factor(as.numeric(data1_test$target2))
levels(target2)[1] <- 0
levels(target2)[2] <- 1
count <- 0
target2 <- as.matrix(target2)
for (i in 1:nrow(fit_values)){
  if ((fit_values[i])==(target2[i])){count <- count + 1}
}
error_rate <- 1 - count/nrow(fit_values)


# 
# # we want to examine the changes in predicted probability associated with one of our two variables
# # We examine the predicted probabilities for each value of minprice (ranging from 40 to 168) at every value of price-min_price (ranging from 40 to 1000) 
# 
# 
# dwrite <- data.frame(priceminprice = rep(c(40:1000), each = 129), minprice = rep(c(40:168),961))
# probability <- predict(test, newdata = dwrite, type = "probs", se = TRUE)
# pp.write <- cbind(dwrite,probability)
# 
# # we want to examine the changes in predicted probability associated with one of our two variables
# # We examine the predicted probabilities for each value of price-min_price (ranging from 40 to 1000) at every value of min_price (ranging from 40 to 168) 
# 
# dwrite <- data.frame(minprice = rep(c(40:168), each = 961), priceminprice = rep(c(40:1000),129))
# probability <- predict(test, newdata = dwrite, type = "probs", se = TRUE)
# pp.write <- cbind(dwrite,probability)


