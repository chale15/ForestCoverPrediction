library(tidyverse)
library(tidymodels)
library(vroom)
library(parsnip)
library(keras)
library(baguette)
library(bonsai)
library(stacks)
library(naivebayes)
library(discrim)

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/ForestCoverPrediction/")

train <- vroom("./From Kaggle/train.csv") %>%
  mutate(Cover_Type=factor(Cover_Type))

test <- vroom("./From Kaggle/test.csv")

folds <- vfold_cv(train, v = 5, repeats = 1)


############################# Random Forest Model ###############################################

my_recipe <- recipe(Cover_Type~., data=train) %>%
  step_impute_median(contains("Soil_Type")) %>%
  step_rm(Id) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

my_recipe1 <- recipe(Cover_Type~., data = train) %>% 
  step_impute_median(contains("Soil_Type")) %>%
  #step_mutate(SoilType = pmax(
  #  Soil_Type1, Soil_Type2 * 2, Soil_Type3 * 3, Soil_Type4 * 4, Soil_Type5 * 5,
  #  Soil_Type6 * 6, Soil_Type7 * 7, Soil_Type8 * 8, Soil_Type9 * 9, Soil_Type10 * 10,
  #  Soil_Type11 * 11, Soil_Type12 * 12, Soil_Type13 * 13, Soil_Type14 * 14, Soil_Type15 * 15,
  #  Soil_Type16 * 16, Soil_Type17 * 17, Soil_Type18 * 18, Soil_Type19 * 19, Soil_Type20 * 20,
  #  Soil_Type21 * 21, Soil_Type22 * 22, Soil_Type23 * 23, Soil_Type24 * 24, Soil_Type25 * 25,
  #  Soil_Type26 * 26, Soil_Type27 * 27, Soil_Type28 * 28, Soil_Type29 * 29, Soil_Type30 * 30,
  #  Soil_Type31 * 31, Soil_Type32 * 32, Soil_Type33 * 33, Soil_Type34 * 34, Soil_Type35 * 35,
  #  Soil_Type36 * 36, Soil_Type37 * 37, Soil_Type38 * 38, Soil_Type39 * 39, Soil_Type40 * 40,
  #  na.rm = FALSE)) %>% 
  step_mutate(WildernessArea = pmax(Wilderness_Area1, Wilderness_Area2 * 2, Wilderness_Area3 * 3, Wilderness_Area4 * 4, na.rm = TRUE)) %>%
  #step_rm(contains("Soil_Type")) %>% 
  step_rm(contains("Wilderness_Area")) %>% 
  step_rm(Id) %>% 
  step_mutate(Total_Distance_To_Hydrology = sqrt(Horizontal_Distance_To_Hydrology**2 + Vertical_Distance_To_Hydrology**2)) %>% 
  step_mutate(Elevation_Vertical_Hydrology = Vertical_Distance_To_Hydrology * Elevation) %>% 
  step_mutate(Hydrology_Fire = Horizontal_Distance_To_Hydrology * Horizontal_Distance_To_Fire_Points) %>% 
  step_mutate(Hydrology_Roadways = Horizontal_Distance_To_Hydrology * Horizontal_Distance_To_Roadways) %>% 
  step_mutate(Roadways_Fire = Horizontal_Distance_To_Roadways * Horizontal_Distance_To_Fire_Points) %>% 
  step_mutate_at(WildernessArea, fn = factor) %>% 
  #step_mutate_at(c(SoilType, WildernessArea), fn = factor) %>% 
  step_zv(all_predictors())

prepped <- prep(my_recipe1)
baked <- bake(prepped, new_data = NULL)

sum(is.na(baked$SoilType))

rf_mod <- rand_forest(min_n = tune(), mtry = tune(), trees = 500) %>%
  set_engine('ranger') %>%
  set_mode('classification')

rf_wf <- workflow() %>%
  add_model(rf_mod) %>%
  add_recipe(my_recipe1)

tuning_grid_rf <- grid_regular(min_n(), mtry(c(11,15)),levels = 3)

cv_results_rf <- rf_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid_rf, 
            metrics = metric_set(accuracy, roc_auc))

best_tune_rf <- cv_results_rf %>% select_best(metric='roc_auc')

final_workflow_rf <- rf_wf %>% 
  finalize_workflow(best_tune_rf) %>% 
  fit(data = train)


rf_preds <- predict(final_workflow_rf, new_data = test, type = "class") %>%
  mutate(Cover_Type = .pred_class) %>%
  mutate(Id = test$Id) %>%
  dplyr::select(Id, Cover_Type)

vroom_write(rf_preds, "rf_preds.csv", delim = ",")


############################# MLP Model ###############################################

nn_recipe <- recipe(Cover_Type~., data = train) %>%
  step_impute_median(contains("Soil_Type")) %>%
  step_rm(Id) %>%
  step_zv(all_predictors()) %>%
  step_range(all_numeric_predictors(), min=0, max=1)

nn_model <- mlp(hidden_units = tune(),
                epochs = 50,
                activation = 'softmax') %>%
  set_engine("keras") %>%
  set_mode("classification")

nn_wf <- workflow() %>%
  add_model(nn_model) %>%
  add_recipe(nn_recipe)

tuning_grid_nn <- grid_regular(hidden_units(range=c(5, 15)), levels = 3)

cv_results_nn <- nn_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid_nn, 
            metrics = metric_set(accuracy, roc_auc))

best_tune_nn <- cv_results_nn %>% select_best(metric='roc_auc')

final_workflow_nn <- nn_wf %>% 
  finalize_workflow(best_tune_nn) %>% 
  fit(data = train)

nn_preds <- predict(final_workflow_nn, 
                    new_data = test,
                    type = 'class') %>%
  mutate(Cover_Type = .pred_class) %>%
  mutate(Id = test$Id) %>%
  dplyr::select(Id, Cover_Type)

vroom_write(nn_preds, "nn_preds2.csv", delim = ",")


############################# Boosted Tree Model ###############################################

boost_recipe <- recipe(Cover_Type~., data=train) %>%
  step_impute_median(contains("Soil_Type")) %>%
  step_rm(Id) %>% 
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

boost_mod <- boost_tree(trees = 500, learn_rate = .01, tree_depth = tune()) %>%
  set_engine('xgboost') %>%
  set_mode('classification')

boost_wf <- workflow() %>%
  add_model(boost_mod) %>%
  add_recipe(boost_recipe)


tuning_grid_boost <- grid_regular(tree_depth(range=c(2,6)), levels = 3)

cv_results_boost <- boost_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid_boost, 
            metrics = metric_set(accuracy, roc_auc))

best_tune <- cv_results_boost %>% select_best(metric='roc_auc')

final_workflow_boost <- boost_wf %>% 
  finalize_workflow(best_tune_boost) %>% 
  fit(data = train)

boost_preds <- predict(final_workflow_boost, 
                    new_data = test,
                    type = 'class') %>%
  mutate(Cover_Type = .pred_class) %>%
  mutate(Id = test$Id) %>%
  dplyr::select(Id, Cover_Type)

vroom_write(boost_preds, "boost_preds2.csv", delim = ",")


############################# Naive Bayes Model ###############################################

my_recipe_nb <- recipe(Cover_Type~., data = train) %>% 
  step_mutate(SoilType = pmax(
    Soil_Type1, Soil_Type2 * 2, Soil_Type3 * 3, Soil_Type4 * 4, Soil_Type5 * 5,
    Soil_Type6 * 6, Soil_Type7 * 7, Soil_Type8 * 8, Soil_Type9 * 9, Soil_Type10 * 10,
    Soil_Type11 * 11, Soil_Type12 * 12, Soil_Type13 * 13, Soil_Type14 * 14, Soil_Type15 * 15,
    Soil_Type16 * 16, Soil_Type17 * 17, Soil_Type18 * 18, Soil_Type19 * 19, Soil_Type20 * 20,
    Soil_Type21 * 21, Soil_Type22 * 22, Soil_Type23 * 23, Soil_Type24 * 24, Soil_Type25 * 25,
    Soil_Type26 * 26, Soil_Type27 * 27, Soil_Type28 * 28, Soil_Type29 * 29, Soil_Type30 * 30,
    Soil_Type31 * 31, Soil_Type32 * 32, Soil_Type33 * 33, Soil_Type34 * 34, Soil_Type35 * 35,
    Soil_Type36 * 36, Soil_Type37 * 37, Soil_Type38 * 38, Soil_Type39 * 39, Soil_Type40 * 40,
    na.rm = TRUE)) %>% 
  step_mutate(WildernessArea = pmax(Wilderness_Area1, Wilderness_Area2 * 2, Wilderness_Area3 * 3, Wilderness_Area4 * 4, na.rm = TRUE)) %>%
  step_rm(contains("Soil_Type")) %>% 
  step_rm(contains("Wilderness_Area")) %>% 
  step_rm(Id) %>% 
  #  step_select(Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, WildernessArea, SoilType) %>% 
  step_mutate_at(c(SoilType, WildernessArea), fn = factor) #%>% 
#  step_lencode_glm(all_nominal_predictors(), outcome = vars(Cover_Type))

nb_model <- naive_Bayes(Laplace=tune(),
                        smoothness=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

nb_workflow <- workflow() %>% 
  add_model(nb_model) %>% 
  add_recipe(my_recipe_nb)

tuning_grid_nb <- grid_regular(Laplace(), smoothness(), levels = 10)


cv_results_nb <- nb_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid_nb, 
            metrics = metric_set(roc_auc, accuracy))

best_tune_nb <- cv_results_nb %>% select_best(metric='accuracy')

final_workflow_nb <- nb_workflow %>% 
  finalize_workflow(best_tune_nb) %>% 
  fit(data = train)

nb_preds <- predict(final_workflow, 
                    new_data = test,
                    type = 'class')

nb_submission <- nb_preds %>% 
  bind_cols(., test) %>% 
  select(Id, .pred_class) %>% 
  rename(Cover_Type = .pred_class) 


vroom_write(x=nb_submission, file="./Submissions/NBPreds4.csv", delim=",")
