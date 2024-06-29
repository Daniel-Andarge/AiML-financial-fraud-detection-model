# Machine Learning-based Fraud Detection for E-commerce and Banking Transactions

Adey Innovations Inc. aims to enhance the detection of fraudulent transactions in the e-commerce and banking sectors. This project focuses on developing advanced machine learning models to identify fraud with high accuracy by analyzing transaction data, creating sophisticated features, and implementing real-time monitoring systems. By improving fraud detection, Adey Innovations Inc. aims to reduce financial losses, bolster transaction security, and build stronger trust with customers and financial institutions. The project entails data preprocessing, feature engineering, model development, evaluation, and deployment, ensuring a comprehensive approach to combating fraud.

## Table of Contents

1. [Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
2. [Model Building and Training](#2-model-building-and-training)
3. [Model Explainability Using SHAP](#3-model-explainability-using-shap)
4. [Model Deployment and API Development](#4-model-deployment-and-api-development)
5. [Contributing](#contributing)
6. [License](#license)

## 1. Exploratory Data Analysis (EDA)

### Univariate Analysis

![featureEng](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/eda/his1.png)

### Bivariate Analysis

### Feature Engineering

![featureEng](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/eda/featured_df.png)

## 2. Model Building and Training

After training and testing six models (three for each dataset), we selected the following models:

#### 2.1 Fraud-IP Dataset - XGBoost Model

![xgboost](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/model-building/xg1.png)
![xgboost2](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/model-building/xg2.png)

#### 2.2 Credit Card Dataset - Logistic Regression with StandardScaler

![lr1](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/model-building/lr1.png)
![lr2](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/model-building/lr2.png)

## 3. Model Explainability Using SHAP

### Summary Plot

<img src="https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/shap-lime/summryPlot.png" alt="summary plot" width="600"/>

### Force Plot

![forceplot](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/shap-lime/forcePlot.png)

## 4. Model Deployment and API Development

### Running the Flask App

![runflask](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/api-docker/run-flask.png)

### Testing the API

![testflask](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/api-docker/test-flask.png)

### Building Docker Image

![build](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/api-docker/build-docker-image.png)

### Running Docker Container

![runflask](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/api-docker/docker-run.png)

### Testing the API from Postman

Generated 3 new instances and sent a request to the fraud detection model api.

![postman](https://github.com/Daniel-Andarge/AiML-financial-fraud-detection-model/blob/main/assets/api-docker/postman_tst.png)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
