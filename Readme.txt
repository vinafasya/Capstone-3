California House Price Prediction 

Project Overview
This project aims to predict housing prices in California using machine learning techniques.
It is part of a capstone project to demonstrate end-to-end data science skills, including data cleaning, feature engineering, modeling, and evaluation.

Objectives
The primary objectives of this project are:
Predict housing prices based on various features such as location, median income, and house age.
Analyze key factors influencing house prices.
Evaluate model performance using appropriate metrics.

Dataset
The dataset used for this project is the California Housing Dataset, which includes:

Features:
Longitude, Latitude
Median income
House median age
Rooms per household, population per household
Bedroom-to-room ratio
Ocean proximity categories
Target: Median house value.

Workflow

Data Cleaning:

Removed outliers and handled missing values.
Normalized and scaled necessary features.
Feature Engineering:

Created new features such as bedroom-to-room ratio and location clusters.

Modeling:

Trained the model using XGBoost Regressor.
Evaluated performance using:
Root Mean Square Error (RMSE)
Mean Absolute Error (MAE)
Mean Absolute Percentage Error (MAPE)
Model Deployment:

Built an interactive web app using Streamlit to predict house prices.
