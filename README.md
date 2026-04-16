# Credit Score Classification --- Deep Learning

## Overview

This project builds a **deep learning based credit scoring system**
that predicts whether a customer belongs to **Poor, Standard, or Good
credit categories** using financial behavior data.

Traditional credit evaluation methods rely heavily on rule‑based or
statistical models. This project demonstrates how **Artificial Neural
Networks (ANNs)** can learn complex relationships in financial data to
produce more accurate and scalable credit risk predictions.

------------------------------------------------------------------------

## Problem Statement

Financial institutions must quickly determine whether an individual is
likely to repay a loan. Traditional scoring approaches may fail to
capture complex relationships between financial variables such as:

-   income
-   debt
-   payment history
-   credit utilization
-   number of accounts and loans

The goal of this project is to build a **multi‑class classification
model** capable of predicting creditworthiness with strong
generalization on unseen data.

Target Classes:

-   `0 → Poor`
-   `1 → Standard`
-   `2 → Good`

------------------------------------------------------------------------

## Dataset

Dataset Source: **Kaggle Credit Score Dataset**

Characteristics:

-   **100,000 rows**
-   **28 original features**
-   Expanded to **37 features after preprocessing**
-   Each customer contains financial information recorded across
    **multiple months**

Key features include:

-   Age
-   Annual Income
-   Monthly Salary
-   Number of Bank Accounts
-   Number of Credit Cards
-   Outstanding Debt
-   Credit Utilization Ratio
-   Credit History Age
-   Payment Behavior
-   Total EMI per Month
-   Monthly Balance

Target variable:

    Credit_Score → {Poor, Standard, Good}

------------------------------------------------------------------------

## Project Architecture

### Data Pipeline

    Raw Dataset
         ↓
    Data Cleaning
         ↓
    Feature Engineering
         ↓
    Encoding & Scaling
         ↓
    Train / Test Split
         ↓
    Neural Network Training
         ↓
    Model Evaluation

------------------------------------------------------------------------

## Data Preprocessing

The preprocessing pipeline addresses real-world financial data
challenges.

### Cleaning Steps

-   Removed irrelevant identifiers (`ID`, `Customer_ID`, `Name`, `SSN`)
-   Filled missing values using **customer-level mode**
-   Outlier handling using **forward fill / backward fill**
-   Converted mixed text/numeric columns to numeric formats

### Feature Engineering

-   One-hot encoding for categorical columns
-   Mapping categorical values to numeric labels
-   Correlation analysis using heatmaps

### Feature Scaling

All features were standardized using **StandardScaler**:

    mean = 0
    std  = 1

This improves convergence during neural network training.

------------------------------------------------------------------------

## Model Development

Multiple ANN architectures were tested to balance:

-   performance
-   generalization
-   stability

### Model Evolution

  Model         Optimizer   Regularization   BatchNorm   Dropout
  ------------- ----------- ---------------- ----------- ---------
  Model 1       SGD         None             No          No
  Model 2       Adam        None             No          No
  Model 3       SGD         L1               Yes         Yes
  Final Model   Adam        L1               Yes         Yes

------------------------------------------------------------------------

## Final Model Architecture

    Input Layer

    Dense(256) + ReLU
    BatchNorm
    Dropout(0.35)

    Dense(512) + ReLU
    BatchNorm
    L1 Regularization

    Dense(256) + ReLU
    BatchNorm
    Dropout(0.10)

    Dense(256) + ReLU
    BatchNorm
    Dropout(0.10)

    Output Layer
    Dense(3) + Softmax

Training configuration:

-   Optimizer: **Adam**
-   Learning Rate: **0.0003**
-   Epochs: **250**
-   Batch Size: **1024**
-   Loss: **Sparse Categorical Crossentropy**

------------------------------------------------------------------------

## Model Performance

Final model results:

  Metric              Score
  ------------------- -----------
  Training Accuracy   **86.4%**
  Test Accuracy       **83.4%**

The model achieves strong generalization with minimal overfitting.

------------------------------------------------------------------------

## Evaluation Metrics

Model performance was evaluated using:

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   Confusion Matrix

Example insights:

-   Strong recall for **Poor credit**
-   High precision for **Standard credit**
-   Balanced performance across all classes

------------------------------------------------------------------------

## Technologies Used

Programming Language:

-   Python

Machine Learning & Deep Learning:

-   TensorFlow
-   Keras
-   Scikit-learn

Data Processing:

-   Pandas
-   NumPy

Visualization:

-   Matplotlib

Development Environment:

-   Google Colab
-   Jupyter Notebook

------------------------------------------------------------------------

## Key Challenges

### Imbalanced Dataset

Certain credit classes had fewer samples.

Solution:

-   Applied **class weighting** during training.

### Data Quality Issues

Dataset contained missing values and inconsistent formats.

Solution:

-   Customer-level imputation
-   Forward/backward filling
-   Categorical normalization

### Overfitting

Initial models memorized training data.

Solution:

-   Dropout layers
-   Batch normalization
-   L1 regularization
