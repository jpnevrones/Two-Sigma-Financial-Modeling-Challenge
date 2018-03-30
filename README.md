# Two-Sigma-Financial-Modeling-Challenge
Two Sigma Financial Modeling Challenge - The goal of the competition is to predict a variable ‘y’ which depends on 110 anonymized features pertaining to a time-varying value for a financial instrument, No further information are provided on the meaning of the features, the transformations that were applied to them, the timescale, or the type of instruments that are included in the data. Prototyped predictive models, using linear regression and extra-trees regressor to develop a predictive model for predicting the output variable ‘y’. 

With the below methods I have held a rank of [554/1578] among the top 36%.

The goal of the competition is to predict a variable ‘y’ which depends on 110 anonymized features pertaining to a time-varying value for a financial instrument, No further information are provided on the meaning of the features, the transformations that were applied to them, the timescale, or the type of instruments that are included in the data. Submissions are evaluated on the R(also called as coefficient of determination) value between the predicted and actual values.

Applied regression analysis to determine the statistical significance of features to be used for the prediction model.
- Performed exploratory data analysis for feature selection and understand the type of relation between inputs instrument feature and output at different levels of hierarchies. Used Basic Structural Time Series Method for time series missing value imputation.
- Prototyped predictive models, using linear regression and extra-trees regressor to develop a predictive model for predicting the output variable ‘y’.
  - [Ridge plus Extra Trees Regressor with moving avaerage to tackle lag](https://github.com/jpnevrones/Two-Sigma-Financial-Modeling-Challenge/blob/master/ExtraTree-plus-LR-Ridge.py)
  - [ExtraTreesRegressor plus Linear Regressor](https://github.com/jpnevrones/Two-Sigma-Financial-Modeling-Challenge/blob/master/ExtraTreesRegressorLR.py)
  - [Ridge Regressor](https://github.com/jpnevrones/Two-Sigma-Financial-Modeling-Challenge/blob/master/Ridge.py)
  - [XGBoost](https://github.com/jpnevrones/Two-Sigma-Financial-Modeling-Challenge/blob/master/XGBOOST-Two-sigma.py)

**Dataset - Provided by 2sigma :** “This dataset contains anonymized features pertaining to a time-varying value for a financial instrument. Each instrument has an id. Time is represented by the ‘timestamp’ feature and the variable to predict is ‘y’. No further information will be provided on the meaning of the features, the transformations that were applied to them, the timescale, or the type of instruments that are included in the data. Moreover, in accordance with competition rules, participants must not use data other than the data linked from the competition website for the purpose of use in this competition to develop and test their models and submissions.”
