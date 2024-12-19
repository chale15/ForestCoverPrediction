# ForestCoverPrediction

Code and Submissions for the Store Item Demand Forecasting Challenge Kaggle Competition


## Contest Description

Random forests? Cover trees? Not so fast, computer nerds. We're talking about the real thing.

In this competition you are asked to predict the forest cover type (the predominant kind of tree cover) from strictly cartographic variables (as opposed to remotely sensed data). The actual forest cover type for a given 30 x 30 meter cell was determined from US Forest Service (USFS) Region 2 Resource Information System data. Independent variables were then derived from data obtained from the US Geological Survey and USFS. The data is in raw form (not scaled) and contains binary columns of data for qualitative independent variables such as wilderness areas and soil type.

This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.


## Submission

Submissions are evaluated on multi-class classification accuracy.




## Model Description

The model with the best performance for this competition was a stacked model, made up of a Random Forest model, a Multi-layer Perceptron (Neural Net), and a Boosted Tree model (XGBoost).

### Top Score: 0.78770

### Underlying Models:

**Model 1: Random Forest**

Top Score: 0.76054

Best Tune: min_n = 2, mtry = 15

Pseudocode Recipe: Impute Soil Type median, Condense Wilderness Area into single-column factor, remove ID and zero-variance predictors, Create ‘Total_Distance_to_Hydrology’, ‘Hydrology_Fire’, ’Elevation_Vertical_Hydrology’, ‘Hydrology_Roadway’, and ‘Roadways_Fire’ features

**Model 2: MLP**

Top Score: 0.52816

Best Tune: hidden_units = 15

Pseudocode Recipe: Impute Soil Type median, remove ID and zero-variance predictors, range numeric predictors from 0 to 1

**Model 3: Boosted Tree**

Top Score: 0.64032

Best Tune: tree_depth = 6

Pseudocode Recipe: Impute Soil Type median, remove ID and zero-variance predictors, normalize numeric predictors



Will Cukierski. Forest Cover Type Prediction. https://kaggle.com/competitions/forest-cover-type-prediction, 2014. Kaggle.
