# Freight-Rate-Prediction-Challenge
Machine learning project for freight rate prediction. Includes data exploration, preprocessing, feature engineering, model training and validation, predictions for 12,000 validation loads, and December 2025 rate forecasting with the provided scoring script.

Project Overview

The objective is to predict the posted_rate of freight loads using historical shipment information such as:

Pickup and delivery locations
Distance
Equipment type
Weight
Date
Route information

Several HistGradientBoostingRegressor configurations are compared using a time-based validation strategy.
The final model is trained on all available development data and used to:

Predict the 12,000 loads in data/validation.csv.
Fill predicted_rate in data/validation_predictions_template.csv.
Save the completed predictions as validation_predictions.csv.
Predict every row in data/december_chart_inputs.csv.
Fill the existing predicted_rate column in the December input file.
Dataset

The project uses the following files:

data/
├── train_test.csv

├── validation.csv

├── validation_predictions_template.csv

└── december_chart_inputs.csv
Development Data

train_test.csv contains the labeled historical data used for model development.
The target variable is:

posted_rate
Validation Data

validation.csv contains unlabeled loads for which predictions must be generated.

Validation Template

validation_predictions_template.csv provides the required format for the final validation submission.

The output file is:

validation_predictions.csv
December Data
december_chart_inputs.csv contains the December loads used to generate the final December rate predictions.

Methodology
1. Data Cleaning

The model checks for:

Duplicate load IDs
Duplicate rows
Missing values

Missing numerical values are replaced using the median, while missing categorical values are replaced using the most frequent category.

2. Feature Engineering
   The model uses:

Pickup
Delivery
Equipment
Distance
Weight
Route
Year
Month
Day of week
Day of month
Week of year
Day of year
Weekend indicator
Log-transformed distance

The route is represented as:

pickup -> delivery
3. Validation Strategy

A chronological split is used to simulate future prediction:

January – September 2025 → Training
October 2025             → Validation

This avoids using future observations to predict earlier observations and better represents the final December forecasting task.

4. Model Comparison

Five HistGradientBoostingRegressor configurations are evaluated.

The models are compared using:

MAE
RMSE
R²

The model with the lowest validation MAE is selected as the final model.

5. Final Training

After selecting the best model, it is retrained using all available labeled development data.

The final model is then used to generate:

validation_predictions.csv

and December predictions.
