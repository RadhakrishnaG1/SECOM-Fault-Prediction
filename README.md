# Semiconductor Wafer Fault Prediction using Neural Networks

Predicting semiconductor wafer failures using manufacturing sensor data from the UCI SECOM dataset. This project explores data preprocessing, neural networks, overfitting analysis, regularization, and threshold optimization for defect detection in semiconductor fabrication processes.

---

## Overview

Semiconductor fabrication plants collect hundreds of sensor measurements during wafer manufacturing. Defects are often detected only after expensive final testing. The objective of this project is to predict whether a wafer will pass or fail final testing using manufacturing sensor data.

This project uses the SECOM manufacturing dataset and develops machine learning models to identify defective wafers before final testing.

---

## Problem Statement

Given hundreds of sensor readings collected during semiconductor manufacturing, predict whether a wafer will:

- Pass final testing (0)
- Fail final testing (1)

Early identification of defective wafers can reduce manufacturing costs and improve production efficiency.

---

## Dataset

**Dataset:** SECOM Manufacturing Dataset

### Dataset Characteristics

| Property | Value |
|-----------|---------|
| Samples | 1567 |
| Features | ~590 Sensor Measurements |
| Target | Pass / Fail |
| Missing Values | Present |
| Class Distribution | Highly Imbalanced |

### Target Labels

| Label | Meaning |
|---------|---------|
| 0 | Pass |
| 1 | Fail |

---

## Data Preprocessing

The following preprocessing steps were performed:

- Removed features with excessive missing values
- Replaced remaining missing values using median imputation
- Applied feature scaling using StandardScaler
- Split data into training and testing sets

---
## How to Run

1. Clone the repository

```bash
git clone <repository-url>
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Open Jupyter Notebook

```bash
jupyter notebook
```

4. Run:

```text
Semiconductor_Wafer_Fault_Prediction.ipynb
```

## Models Implemented

### 1. Logistic Regression

Used as a baseline model for comparison.

### 2. Initial Neural Network

Architecture:

```text
Input Features
      ↓
Dense(128, ReLU)
      ↓
Dense(64, ReLU)
      ↓
Dense(1, Sigmoid)
```

### Observation

The model achieved high overall accuracy but performed poorly on defective-wafer detection and showed signs of overfitting.

---

### 3. Improved Neural Network

Architecture:

```text
Input Features
      ↓
Dense(32, ReLU)
      ↓
Dense(16, ReLU)
      ↓
Dense(1, Sigmoid)
```

Improvements:

- L2 Regularization
- Early Stopping

These techniques were used to reduce overfitting and improve generalization.

---

## Class Distribution

![Class Distribution](images/class_distribution.png)

The dataset is highly imbalanced, with significantly more passing wafers than failing wafers.

---

## Initial Neural Network (Before Regularization)

![Before Fix](images/nn_before_fix.png)

### Observation

Training loss decreased rapidly toward zero while validation loss increased steadily.

This indicates severe overfitting, where the model memorized the training data but failed to generalize to unseen examples.

---

## Regularized Neural Network

![After Fix](images/nn_after_fix.png)

### Observation

L2 regularization and early stopping reduced overfitting and improved generalization performance.

---

## Model Comparison

| Model | Accuracy | Precision (Fail) | Recall (Fail) | F1-Score (Fail) |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | `0.84` | `0.11` | `0.19` | `0.14` |
| **Initial Neural Network (Unregularized)** | `0.92` | `0.17` | `0.05` | `0.07` |
| **Improved Neural Network (Threshold = 0.05)** | `0.43` | `0.09` | **`0.81`** | **`0.16`** |

### Key Observation

While the baseline neural network showed a high overall accuracy of `0.92`, it failed on the minority class, only catching **5%** of actual defects (Recall = 0.05). This highlights why global accuracy is a misleading metric for highly imbalanced datasets. 

By applying L2 regularization, early stopping, and shifting the decision threshold to **0.05**, the model's defect detection rate increased to **81%**. In a manufacturing context, trading off overall accuracy to maximize recall is preferred because the cost of missing a defective wafer is significantly higher than managing a false alarm.

---

## Threshold Optimization

The default classification threshold of 0.5 resulted in poor failure detection.

Tested multiple classification thresholds to evaluate the trade-off between precision and recall.

| Threshold | Precision (Fail) | Recall (Fail) | F1-Score (Fail) | Operational Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **0.05** | `0.09` | **`0.81`** | **`0.16`** | Selected. Detects 17 out of 21 defects. |
| **0.10** | `0.08` | `0.38` | `0.14` | Detects 8 out of 21 defects. |
| **0.15** | `0.11` | `0.24` | `0.15` | Detects 5 out of 21 defects. |
| **0.20** | `0.10` | `0.10` | `0.10` | Detects 2 out of 21 defects. |
| **0.25** | `0.17` | `0.10` | `0.12` | Detects 2 out of 21 defects. |

### Selected Threshold

**Threshold = 0.05** For the final deployment, **0.05** was selected as the optimal operational threshold. At this setting, the model achieves its highest failure-class F1-score (`0.16`) and a powerful **Recall of `0.81`**, successfully flagging 17 out of the 21 actual defective wafers in the test set. 

In a semiconductor fabrication environment, missing a defective wafer costs far more than managing false alarms. Lowering the threshold to 0.05 prioritizes finding faults early to avoid passing defective products through final quality control.
## Key Findings

- Semiconductor manufacturing data contains many missing values and highly imbalanced labels.
- Accuracy alone is not a reliable metric for evaluating defect-detection systems.
- Neural networks can overfit small industrial datasets.
- L2 regularization and early stopping improved model generalization.
- Threshold tuning significantly improved defective-wafer detection performance.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- TensorFlow / Keras
- Jupyter Notebook

---

## Repository Structure

```text
SECOM-Fault-Prediction/

README.md
requirements.txt
Semiconductor_Wafer_Fault_Prediction.ipynb

images/
├── class_distribution.png
├── nn_before_fix.png
└── nn_after_fix.png
```
---

## Future Improvements

Potential future work includes:

- Feature selection techniques
- Random Forests and Gradient Boosting
- More advanced neural network architectures
- Explainable AI methods for identifying important manufacturing sensors

---

## Author

**Radhakrishna G**  
Engineering Physics  
Indian Institute of Technology Madras
