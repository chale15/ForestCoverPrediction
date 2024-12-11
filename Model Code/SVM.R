library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(workflows)
library(glmnet)
library(naivebayes)
library(ranger)
library(kknn)
library(discrim)
library(kernlab)
library(themis)

#Read Data

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/ForestCoverPrediction/")
#setwd("~/Kaggle/Forests")

train <- vroom("./From Kaggle/train.csv")
test <- vroom("./From Kaggle/test.csv")

train$Cover_Type <- as.factor(train$Cover_Type)


my_recipe <- recipe(Cover_Type~., data = train) %>%
  step_mutate(WildernessArea = pmax(Wilderness_Area1, Wilderness_Area2 * 2, Wilderness_Area3 * 3, Wilderness_Area4 * 4, na.rm = TRUE)) %>%
  
  step_mutate_at(color, fn = factor) %>% 
  step_mutate(id, feature = id) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>% 
  step_smote(all_outcomes(), neighbors=3) %>% 
  step_range(all_numeric_predictors(), min=0, max=1)

my_recipe <- recipe(Cover_Type~., data = train) %>% 
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
  step_mutate(Total_Distance_To_Hydrology = sqrt(Horizontal_Distance_To_Hydrology**2 + Vertical_Distance_To_Hydrology**2)) %>% 
  step_mutate(Elevation_Vertical_Hydrology = Vertical_Distance_To_Hydrology * Elevation) %>% 
  step_mutate(Hydrology_Fire = Horizontal_Distance_To_Hydrology * Horizontal_Distance_To_Fire_Points) %>% 
  step_mutate(Hydrology_Roadways = Horizontal_Distance_To_Hydrology * Horizontal_Distance_To_Roadways) %>% 
  step_mutate(Roadways_Fire = Horizontal_Distance_To_Roadways * Horizontal_Distance_To_Fire_Points) %>% 
  step_mutate_at(c(SoilType, WildernessArea), fn = factor) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  #step_lencode_mixed(all_nominal_predictors(), outcome = vars(Cover_Type)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_pca(all_predictors(), threshold=0.9)

prepped <- prep(my_recipe)
baked <- bake(prepped, new_data = NULL)

#SVM Models

svm_radial <- svm_rbf(rbf_sigma=tune(),
                      cost=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kernlab")

radial_workflow <- workflow() %>% 
  add_model(svm_radial) %>% 
  add_recipe(my_recipe)

tuning_grid_radial <- grid_regular(rbf_sigma(), cost(), levels = 5)

folds <- vfold_cv(train, v = 10, repeats=1)

cv_results_radial <- radial_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid_radial, 
            metrics = metric_set(accuracy, roc_auc))


best_tune_radial <- cv_results_radial %>% select_best(metric='accuracy')


final_workflow_radial <- radial_workflow %>% 
  finalize_workflow(best_tune_radial) %>% 
  fit(data = train)

radial_svm_preds <- predict(final_workflow_radial, 
                            new_data = test,
                            type = 'class')


radial_svm_submission <- radial_svm_preds %>% 
  bind_cols(., test) %>% 
  select(Id, .pred_1) %>% 
  rename(Cover_Type = .pred_1) 

vroom_write(x=radial_svm_submission, file="./Submissions/RadialSVMpreds1.csv", delim=",")
