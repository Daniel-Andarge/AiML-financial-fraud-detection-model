# Machine Learning-based Fraud Detection for E-commerce and Banking Transactions

This project aims to significantly enhance the identification of fraudulent activities within E-commerce and banking sectors. It focuses on developing advanced machine learning models that analyze transaction data, employ sophisticated feature engineering techniques, and implement real-time monitoring systems to achieve high accuracy in fraud detection.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Univariate Analysis](#univariate-analysis)
   - [Bivariate Analysis](#bivariate-analysis)
4. [Feature Engineering](#feature-engineering)
5. [Model Building and Training](#model-building-and-training)
   - [Fraud-IP Dataset - XGBoost Model](#fraud-ip-dataset---xgboost-model)
   - [Credit Card Dataset - Logistic Regression with StandardScaler](#credit-card-dataset---logistic-regression-with-standardscaler)
6. [Model Explainability Using SHAP](#model-explainability-using-shap)
   - [Summary Plot](#summary-plot)
   - [Force Plot](#force-plot)
7. [Model Deployment and API Development](#model-deployment-and-api-development)
   - [Running the Flask App](#running-the-flask-app)
   - [Testing the API](#testing-the-api)
   - [Building Docker Image](#building-docker-image)
   - [Running Docker Container](#running-docker-container)
   - [Testing the API from Postman](#testing-the-api-from-postman)
8. [Project Report](#project-report)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview

This project aims to significantly improve the identification of fraudulent activities within these sectors. It focuses on developing advanced machine learning models that analyze transaction data, employ sophisticated feature engineering techniques, and implement real-time monitoring systems to achieve high accuracy in fraud detection.

## Data Collection and Preprocessing

Gather and preprocess transaction data to ensure it is clean and usable for analysis. This includes data cleaning, handling missing values, and normalization.

## Exploratory Data Analysis (EDA)

Analyze customer transaction characteristics to identify patterns and trends influencing fraud detection.

### Univariate Analysis

![Univariate Analysis](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/eda/his1.png)

### Bivariate Analysis

For detailed insights and visualizations related to bivariate analysis, please refer to the [EDA Notebook](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/notebooks/eda.ipynb).


## Feature Engineering

Create new features that enhance the predictive power of the models based on insights from EDA.

![Feature Engineering](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/eda/featured_df.png)

## Model Building and Training

After training and testing multiple models, we selected the following:

### Fraud-IP Dataset - XGBoost Model

![XGBoost Model](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/model-building/xg1.png)
![XGBoost Model Evaluation](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/model-building/xg2.png)

### Credit Card Dataset - Logistic Regression with StandardScaler

![Logistic Regression Model](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/model-building/lr1.png)
![Logistic Regression Model Evaluation](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/model-building/lr2.png)

## Model Explainability Using SHAP

### Summary Plot

<img src="https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/shap-lime/summryPlot.png" alt="Summary Plot" width="600"/>

### Force Plot

![Force Plot](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/shap-lime/forcePlot.png)

## Model Deployment and API Development

### Running the Flask App

![Running Flask App](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/api-docker/run-flask.png)

### Testing the API

![Testing the API](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/api-docker/test-flask.png)

### Building Docker Image

![Building Docker Image](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/api-docker/build-docker-image.png)

### Running Docker Container

![Running Docker Container](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/api-docker/docker-run.png)

### Testing the API from Postman

Generated new instances and sent requests to the fraud detection model API.

![Postman Testing](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/api-docker/postman_tst.png)

## Project Report

For a comprehensive overview of the project, please refer to the project report: [Project Report PDF](https://drive.google.com/file/d/1QaTrq0ID5fQkPboBedNT7lTdlgZ6Pkme/view).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
