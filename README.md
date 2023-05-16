# TimeSeries_ElectricLoad

# Electricity Load and Weather Analysis in Taiwan

This repository contains the code and data used for analyzing the electricity load and weather patterns in Taiwan. The analysis aims to investigate the relationship between electricity load and weather factors, and predict future electricity development in Taiwan.

## Abstract

The abstract provides an overview of the project and its objectives, including the use of a shallow learning model for prediction and the potential implications for energy allocation in Taiwan.

## Introduction

The introduction section introduces the project and outlines the various steps involved, including data preprocessing, time series decomposition, model selection, and prediction.

## Dataset Description

The dataset used in this project is sourced from Kaggle and includes electricity load data and weather data for Taiwan from 2017 to 2021. The dataset consists of several variables, including time series data, categorical variables, and numerical variables related to weather and climate factors.

Time versus load

<img src="img/Load.png" width="400">

ACF/PACF of the dependent variable

<img src="img/AR.png" width="400">

Correlation matrix

<img src="img/CM.png" width="400">


## Preprocessing

The preprocessing section describes the steps taken to clean and prepare the dataset for analysis, including handling missing values, filling data based on distribution, and splitting the dataset into training and testing sets.

## Stationarity

Rolling mean and variance of load of North

<img src="img/RM.png" width="400">

The stationarity section examines the stationarity of the data using ADF and KPSS tests, as well as rolling mean and variance analysis.

## Time Series Decomposition

The result of trending and seasonality of the original data

<img src="img/Decomposition.png" width="400">

The time series decomposition section applies the STL (Seasonal-Trend decomposition using Loess) method to analyze the trending and seasonality components of the data.

## Holt-Winters Method

<img src="img/HW.png" width="400">

The Holt-Winters method is implemented to forecast electricity load using the train dataset and evaluate its performance against the test dataset.

## Feature Selection

The feature selection section discusses the process of selecting relevant features for the prediction model, including checking for collinearity and performing backward stepwise regression.

## Base Models

Several base models, including AFM, Naïve, Drift, and SES, are implemented to establish baseline predictions for electricity load.

## Multiple Linear Regression
A multiple linear regression model is developed using the OLS (Ordinary Least Squares) method and evaluated based on statistical criteria and prediction accuracy.

One-step prediction of model

<img src="img/Onestep.png" width="400">

H-step prediction of model

<img src="img/Hstep.png" width="400">

## ARMA, ARIMA, and SARIMA Model

ARMA, ARIMA, and SARIMA models are tested to analyze the stationarity of the data and select the best-fitting model for prediction.

By observing the ACF/PACF plot that generated in the previous section, through ACF I’ll guess that there might be a 24 lags period of the data, and by PACF I’ll guess there might be an order of AR.

GPAC table of original data

<img src="img/GPAC.png" width="400">

## Levenberg Marquardt Algorithm

The Levenberg Marquardt algorithm is applied to fit the data into the selected ARIMA model, and diagnostic tests are conducted to assess the model's performance.


## Diagnostic Analysis

The diagnostic analysis section includes various tests and assessments, such as confidence intervals, zero/pole cancellation, chi-square tests, and analysis of residual variance.

## Deep Learning Model

<img src="img/LSTM.png" width="400">

A LSTM (Long Short-Term Memory) deep learning model is implemented to forecast electricity load, and its performance is evaluated based on training and testing scores.

## Final Model Selection

The best-performing models, including LSTM and SARIMA, are compared based on their Mean Squared Error (MSE) values, and the LSTM model is selected as the final model.

## Forecast Function

The forecast function section presents the final forecast function derived from the selected ARIMA model.

For the forecast function, I’m using the ARIMA(24,1,0) model, which is the best model with a presentable forecast function. 

Function:
𝑦(𝑡) – 0.16*𝑦(𝑡 − 1) – 0.5*𝑦(𝑡 − 24)= 𝑒(𝑡)


## h-step Ahead Predictions

SARIMA model prediction

<img src="img/SARIMA.png" width="400">

The h-step ahead predictions section showcases the predictions made by the selected model compared to the actual test data.



## Summary and Conclusion

For the final model selected in this project is a SARIMA model, which only fit with time series data but can’t predict using other features. Although the original goal of this project is to find out if the weather data is connected to the electricity load, the performance of pure time series prediction using SARIMA model and LSMT is better. 
On the other hand, the LSMT deep learning model performed better but the training time takes much longer than traditional time series model like SARIMA. Although the error is smaller than traditional model but if we discuss about efficiency, I will say that traditional model is better for dataset that is too large. However, if you get enough time and ram for training model, deep learning model like LSMT will be a better choice.


## Required Packages

The following packages need to be imported for the code to run correctly:

### Data Analysis and Visualization

- pandas: `import pandas as pd`
- matplotlib: `import matplotlib.pyplot as plt`
- numpy: `import numpy as np`
- seaborn: `import seaborn as sns`

### Time Series Analysis

- statsmodels.tsa.stattools: 
  - adfuller: `from statsmodels.tsa.stattools import adfuller`
  - kpss: `from statsmodels.tsa.stattools import kpss`
- statsmodels.graphics.tsaplots: 
  - plot_acf: `from statsmodels.graphics.tsaplots import plot_acf`
  - plot_pacf: `from statsmodels.graphics.tsaplots import plot_pacf`
- scipy.stats: `from scipy.stats import chi2`
- statsmodels.api: `import statsmodels.api as sm`
- statsmodels.tsa.seasonal: `from statsmodels.tsa.seasonal import STL`
- statsmodels.tsa.holtwinters: `import statsmodels.tsa.holtwinters as ets`

### Feature Selection

- numpy.linalg: `from numpy import linalg as LA`
- sklearn.ensemble.RandomForestClassifier: `from sklearn.ensemble import RandomForestClassifier`
- sklearn.feature_selection.SelectFromModel: `from sklearn.feature_selection import SelectFromModel`
- sklearn.ensemble.RandomForestRegressor: `from sklearn.ensemble import RandomForestRegressor`
- sklearn.model_selection.train_test_split: `from sklearn.model_selection import train_test_split`

### LSTM

- numpy: `import numpy`
- pandas: `import pandas`
- math: `import math`
- keras.models.Sequential: `from keras.models import Sequential`
- keras.layers.Dense: `from keras.layers import Dense`
- keras.layers.LSTM: `from keras.layers import LSTM`
- sklearn.preprocessing.MinMaxScaler: `from sklearn.preprocessing import MinMaxScaler`
- sklearn.metrics.mean_squared_error: `from sklearn.metrics import mean_squared_error`




