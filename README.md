# Project 2 - Ames Housing Data and Housing Price Predictions


## 1. Problem:
Create a housing price model that will predict housing sale prices based on the features in the Ames Housing Dataset.

The data description can be found at : http://jse.amstat.org/v19n3/decock/DataDocumentation.txt.

## 2. Executive Summary
The data was imported via pandas, and checked for null values. A basic OLS regression was performed on scaled, intact, continuous numeric columns. Correlations were generated, and the most promising coefficients were selected. From there, a basic model was generated from only the intact numeric columns. Columns with small amounts of NaN values were imputes with the column average.

At this point, a polynomial expansion of the numeric features was generated and tested via a LASSO optimization. The LASSO model generated lower variance in the predictions, and the non-zero coefficients were noted and used on the next iteration of the model.

Dummy features were then explored, starting with the ones showing the highest correlation. Iteratively adding these yielded a final R2 score of approximately 0.900 for both training and test data. The predictions for the Kaggle test were then uploaded.

## 3. Exploratory Data Analysis
Several columns were found to have high amounts of NaNs. For the most part, these columns were either ordinal, discrete, or nominal, which had little effect on the data analysis. For those columns with small NaN that were continuous, values were imputed with the column mean. For columns that were continuous with high NaN, the columns were simply dropped.

## 4. Initial Modeling
We started with a model that only incorporates the numerical features that we have selected according to correlation score and submitted to Kaggle to see how we did relative to the rest of the cohorts. Because this data set has so many features, we did our subsequent EDA and modeling a little more piecemeal as we refined the model.

The next step was to do a LassoCV regression of the polynomial expansion of the original features. From the polynomial expansion, the features with coefficients >0 were selected for further refinement. Those features were then input into a standard Linear Regression, and the dummy features that were chosen were added one by one. Charts and metrics were generated for each addition, and features that led to improved metrics were saved.

## 5. Conlclusions:
A reasonable model of the data was generated from approximately 15 of the polynomial expansion features, as well as approximately 10 of the dummy features. The R2 scores for both training and testing data showed excellent parity (~.90 for training and ~0.89 for testing). Cross val scores (K=10) were approximately 0.87 for the training data.
