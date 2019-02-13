# Stochastic Optimization

## Purpose 

The purpose of this repository is to learn the basics of contemporary stochastic optimization methods.
Inference for linear regression coefficients via stochastic optimization has been conducted for three different datasets.

## Algorithms studied
The stochastic optimization algorithms considered are Stochastic Gradient Descent (SGD), Stochastic Gradient Descent with Momentum (MSGD), 
Stochastic Gradient Descent with Nesterov Accelerated Gradient (NAGSGD), AdaGrad, RMSProp and Adam.
I have used linear regression instead of deep learning models in order to understand these optimization algorithms. 

## Dataset Description

The first dataset contains 1000 simulated data points and is associated with two regression coefficients, 
namely the intercept and slope of a simple linear regression model. 
Starting with simple linear regression and simulated data, I have developed the involved stochastic optimization algorithms. 
The second dataset is available on Kaggle, pertains to weather in Szegen between 2006-2016 and it contains 96,453 observations and six
parameters. The dataset weather_data.csv can be found on Kaggle at https://www.kaggle.com/budincsevity/szeged-weather.
Note that I have altered the original column labels to make R coding more user-friendly.
The third dataset is the million song dataset (MSD), which is available on the UCI machine learning repository. 
Recall that MSD is a big dataset, containing 515,345 data points and 91 parameters. 
More information about this subset of the million song data set is available at
http://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
