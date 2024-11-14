library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(workflows)

#Read Data

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/ForestCoverPrediction/")
#setwd("~/Kaggle/Forests")

train <- vroom("./From Kaggle/train.csv")
test <- vroom("./From Kaggle/test.csv")

