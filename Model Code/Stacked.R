library(tidyverse)
library(tidymodels)
library(vroom)
library(parsnip)
library(keras)
library(baguette)
library(bonsai)
library(stacks)

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/ForestCoverPrediction/")

train <- vroom("./From Kaggle/train.csv") %>%
  mutate(Cover_Type=factor(Cover_Type))

test <- vroom("./From Kaggle/test.csv")

untuned_model <- control_stack_grid()
tuned_model <- control_stack_resamples()

folds <- vfold_cv(train, v = 5, repeats = 1)

my_recipe <- recipe(Cover_Type~., data=train) %>%
  step_impute_median(contains("Soil_Type")) %>%
  step_rm(Id) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

rf_mod <- rand_forest(min_n = 1, mtry = 15, trees = 500) %>%
  set_engine('ranger') %>%
  set_mode('classification')

rf_wf <- workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(my_recipe)

rf_model <- fit_resamples(rf_wf,
                          resamples = folds,
                          metrics = metric_set(roc_auc),
                          control = tuned_model)


nn_recipe <- recipe(Cover_Type~., data = train) %>%
  step_impute_median(contains("Soil_Type")) %>%
  step_rm(Id) %>%
  step_zv(all_predictors()) %>%
  step_range(all_numeric_predictors(), min=0, max=1)

nn_model <- mlp(hidden_units = 10,
                epochs = 50) %>%
  set_engine("keras") %>%
  set_mode("classification")

nn_wf <- workflow() %>%
  add_model(nn_model) %>%
  add_recipe(nn_recipe)

nn_model <- fit_resamples(nn_wf,
                          resamples = folds,
                          metrics = metric_set(roc_auc),
                          control = tuned_model)

boost_recipe <- recipe(Cover_Type~., data=train) %>%
  step_impute_median(contains("Soil_Type")) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

boost_mod <- boost_tree(trees = 500, learn_rate = .01, tree_depth = 2) %>%
  set_engine('xgboost') %>%
  set_mode('classification')

boost_wf <- workflow() %>%
  add_model(boost_mod) %>%
  add_recipe(boost_recipe)


boost_model <- fit_resamples(boost_wf,
                             resamples = folds,
                             metrics = metric_set(roc_auc),
                             control = tuned_model)

my_stack <- stacks() %>%
  add_candidates(rf_model) %>%
  add_candidates(nn_model) %>%
  add_candidates(boost_model)

stack_mod <- my_stack %>%
  blend_predictions() %>%
  fit_members()

stack_preds <- stack_mod %>%
  predict(new_data = test, type = "class") %>%
  mutate(Cover_Type = .pred_class) %>%
  mutate(Id = test$Id) %>%
  dplyr::select(Id, Cover_Type)
  

vroom_write(x=stack_preds, file="./Submissions/stack_preds4.csv", delim=",")













