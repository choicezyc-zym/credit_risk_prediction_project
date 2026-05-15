# Credit Risk Prediction Project

This project is an end-to-end machine learning classification project for predicting credit default risk using a credit risk dataset from Kaggle.

The goal is to identify high-risk loan applicants based on applicant profile, income, employment information, loan characteristics, and credit history.

---

## Project Goal

Credit risk prediction helps financial institutions identify applicants who may have a higher probability of loan default.

This project follows a practical machine learning workflow:

```text
Business Understanding
↓
Data Loading
↓
Exploratory Data Analysis
↓
Missing Value Handling
↓
Feature Engineering
↓
Model Training
↓
Model Evaluation
↓
Feature Importance Analysis
↓
Single Applicant Risk Prediction
```

---

## Dataset

Dataset used:

```text
Credit Risk Dataset
```

Dataset shape:

```text
32,581 rows
12 columns
```

Target variable:

```text
loan_status
```

Target meaning:

```text
0 = Low Risk / Non-default
1 = High Risk / Default
```

Main features include:

```text
person_age
person_income
person_home_ownership
person_emp_length
loan_intent
loan_grade
loan_amnt
loan_int_rate
loan_percent_income
cb_person_default_on_file
cb_person_cred_hist_length
```

---

## Tech Stack

```text
Python
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

---

## Project Structure

```text
credit_risk_prediction_project/
├── data/
│   └── credit_risk.csv
├── outputs/
│   ├── metrics.json
│   ├── model.pkl
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── random_forest_feature_importance.csv
│   └── random_forest_feature_importance.png
├── src/
│   ├── eda.py
│   ├── train.py
│   └── predict.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Exploratory Data Analysis

The dataset contains 32,581 loan records.

The target distribution is:

```text
Low Risk / Non-default: 78.18%
High Risk / Default:   21.82%
```

This means the dataset is moderately imbalanced, so accuracy alone is not enough for model evaluation.

### Missing Values

Two columns contain missing values:

```text
loan_int_rate:       3,116 missing values
person_emp_length:     895 missing values
```

These missing values were handled during model training using median imputation for numerical features.

### Key EDA Findings

#### 1. Default applicants have lower average income

```text
Average income of low-risk applicants: 70,804.36
Average income of high-risk applicants: 49,125.65
```

Applicants with lower income tend to have higher default risk.

#### 2. High-risk applicants have higher loan-to-income ratio

```text
Average loan_percent_income of low-risk applicants: 0.1488
Average loan_percent_income of high-risk applicants: 0.2469
```

A higher loan amount relative to income is associated with higher default risk.

#### 3. Home ownership is related to default risk

Default rate by home ownership:

```text
RENT:      31.57%
OTHER:     30.84%
MORTGAGE:  12.57%
OWN:        7.47%
```

Applicants who rent have a higher default rate than applicants who own a home or have a mortgage.

#### 4. Loan intent affects default risk

Default rate by loan intent:

```text
DEBTCONSOLIDATION: 28.59%
MEDICAL:           26.70%
HOMEIMPROVEMENT:   26.10%
PERSONAL:          19.89%
EDUCATION:         17.22%
VENTURE:           14.81%
```

Debt consolidation, medical, and home improvement loans show higher default rates.

#### 5. Loan grade is a strong risk indicator

Default rate by loan grade:

```text
G: 98.44%
F: 70.54%
E: 64.42%
D: 59.05%
C: 20.73%
B: 16.28%
A:  9.96%
```

Worse loan grades are strongly associated with higher default risk.

#### 6. Previous default record increases risk

Default rate by previous default record:

```text
Previous default = Y: 37.81%
Previous default = N: 18.39%
```

Applicants with a previous default record are more likely to default again.

---

## Data Preprocessing

The preprocessing pipeline includes:

### Numerical Features

Numerical features were processed with:

```text
Median imputation
StandardScaler
```

### Categorical Features

Categorical features were processed with:

```text
Most frequent value imputation
OneHotEncoder
```

The preprocessing and model training steps were combined using a scikit-learn `Pipeline`.

This makes the workflow cleaner and helps avoid data leakage.

---

## Models

Two models were trained and compared:

```text
Logistic Regression
Random Forest
```

Logistic Regression was used as an interpretable baseline model.

Random Forest was used as a stronger non-linear model.

Both models used `class_weight="balanced"` to reduce the impact of class imbalance.

---

## Model Evaluation

### Logistic Regression

```text
Accuracy:  0.8136
Precision: 0.5515
Recall:    0.7799
F1-score:  0.6461
ROC-AUC:   0.8712
```

### Random Forest

```text
Accuracy:  0.9342
Precision: 0.9779
Recall:    0.7145
F1-score:  0.8257
ROC-AUC:   0.9316
```

Random Forest was selected as the final model because it achieved much stronger overall performance, especially in accuracy, precision, F1-score, and ROC-AUC.

---

## Why Accuracy Is Not Enough

The target classes are imbalanced because high-risk applicants account for about 21.82% of the dataset.

A model can achieve high accuracy by predicting most applicants as low risk, but this may fail to detect real high-risk applicants.

Therefore, this project evaluates models using:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
```

---

## Feature Importance

The most important Random Forest features include:

```text
loan_percent_income
person_income
loan_int_rate
loan_amnt
person_emp_length
loan_grade_D
person_age
person_home_ownership_RENT
cb_person_cred_hist_length
```

Top feature importance values:

```text
loan_percent_income:        0.1988
person_income:              0.1592
loan_int_rate:              0.1281
loan_amnt:                  0.0815
person_emp_length:          0.0541
loan_grade_D:               0.0539
person_age:                 0.0489
person_home_ownership_RENT: 0.0408
```

Interpretation:

- A higher loan-to-income ratio is the strongest risk factor.
- Applicant income is highly important.
- Higher loan interest rates are associated with higher risk.
- Loan amount and employment length also affect risk.
- Loan grade and home ownership provide useful risk signals.

---

## Single Applicant Prediction

A sample high-risk applicant was tested using the saved model.

Sample applicant characteristics:

```text
Age: 24
Income: 30,000
Home ownership: RENT
Employment length: 1 year
Loan intent: DEBTCONSOLIDATION
Loan grade: D
Loan amount: 12,000
Loan interest rate: 16.5
Loan percent income: 0.40
Previous default record: Y
Credit history length: 3
```

Prediction result:

```text
Predicted Risk: High Risk
Default Probability: 1.0000
Risk Level: High
```

This applicant has multiple high-risk characteristics, including low income, high loan-to-income ratio, rent housing status, high interest rate, worse loan grade, and previous default record.

---

## Outputs

The project generates the following outputs:

```text
outputs/model.pkl
outputs/metrics.json
outputs/confusion_matrix_logistic_regression.png
outputs/confusion_matrix_random_forest.png
outputs/random_forest_feature_importance.csv
outputs/random_forest_feature_importance.png
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run EDA

```bash
python src/eda.py
```

### 3. Train models

```bash
python src/train.py
```

### 4. Predict one sample applicant

```bash
python src/predict.py
```

---

## What I Learned

Through this project, I practiced how to complete a credit risk machine learning workflow:

- Understand a financial risk prediction problem
- Inspect tabular credit risk data
- Analyze target imbalance
- Handle missing values
- Encode categorical features
- Standardize numerical features
- Train Logistic Regression and Random Forest models
- Evaluate models using precision, recall, F1-score, ROC-AUC, and confusion matrix
- Analyze feature importance
- Save and reuse a trained model for individual applicant prediction

---

## Business Summary

This project shows that credit default risk is strongly related to loan-to-income ratio, applicant income, loan interest rate, loan amount, employment length, loan grade, and home ownership.

The Random Forest model achieved strong overall performance and was selected as the final model.

From a business perspective, the model can help identify high-risk loan applicants earlier and support more informed credit risk decisions.
