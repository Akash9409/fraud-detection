# Problem Statement and Scope

## Problem
Financial institutions must detect fraudulent card transactions quickly while
minimizing false positives for legitimate customers. The data is highly
imbalanced: 492 fraud cases out of 284,807 transactions (0.173%).

## Objective
Build a fraud detection pipeline that predicts fraud probability, evaluate it
with imbalance-appropriate metrics, and serve predictions through a FastAPI
scoring API.

## Dataset
Kaggle Credit Card Fraud Detection (mlg-ulb/creditcardfraud): 284,807 rows;
columns Time, V1-V28 (anonymized PCA features), Amount, Class.

## In Scope
- EDA on imbalance, feature scaling, and Amount patterns
- Baseline models (Logistic Regression, Random Forest) and a Keras MLP
- Imbalance handling: class weights, SMOTE, threshold tuning
- FastAPI scoring endpoint with sample JSON payload

## Out of Scope
- Real-time streaming architecture
- Personally identifiable information
- Production fraud decisioning

## Evaluation Metrics
PR-AUC (primary), recall, precision, F1, ROC-AUC, confusion matrix,
false-positive cost discussion, inference latency. Accuracy is not reported
as a headline metric because it is misleading at this imbalance.

## Responsible Use
The system supports fraud analysts and does not make automated adverse
decisions without human review.

## Author
<your name> (solo project; covers all roles: lead, data, modeling,
evaluation, deployment)