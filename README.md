# Sampling Assignment – Credit Card Fraud Detection

## Project Overview

This project focuses on understanding how sampling techniques help handle class imbalance in machine learning problems. The dataset used represents credit card transactions, where fraudulent cases are extremely rare compared to normal transactions.

Since most machine learning models assume balanced data, training directly on such datasets leads to biased predictions. To overcome this, different sampling techniques were applied and their impact on various machine learning models was analyzed.

---

## Objective

The objectives of this assignment are:

- To study the effect of class imbalance on model performance
- To balance the dataset using multiple sampling techniques
- To train different machine learning models on sampled data
- To analyze which sampling technique works best for which model

---

## Dataset Description

Dataset: Creditcard_data.csv  
Target Variable: Class  

- 0 → Legitimate transaction  
- 1 → Fraudulent transaction  

The dataset is highly imbalanced, with fraudulent transactions forming a very small percentage of the total data. This makes sampling a necessary preprocessing step.

---

## Sampling Techniques Used

Five different probabilistic sampling strategies were applied:

1. Sampling1 — Simple Random Sampling  
   Randomly selects data points from the balanced dataset.

2. Sampling2 — Systematic Sampling  
   Selects samples at regular intervals from the dataset.

3. Sampling3 — Stratified Sampling  
   Preserves class proportions while sampling.

4. Sampling4 — Cluster Sampling  
   Samples groups of observations based on class clusters.

5. Sampling5 — Bootstrap Sampling  
   Generates samples with replacement to create variability.

---

## Machine Learning Models Applied

Each sampled dataset was trained using the following models:

| Model ID | Algorithm |
|----------|-----------|
| M1       | Logistic Regression |
| M2       | Decision Tree |
| M3       | Random Forest |
| M4       | K-Nearest Neighbors |
| M5       | Support Vector Machine |

---

## Methodology

The project was carried out through a structured machine learning pipeline consisting of data preparation, sampling, model training, and evaluation. The steps followed are described below.

### 1. Data Loading and Exploration

The credit card dataset was imported using the Pandas library. Basic inspection was performed to understand the structure of the dataset, identify the target variable, and examine class distribution. The dataset was found to be highly imbalanced, with fraudulent transactions forming a very small minority.

### 2. Handling Class Imbalance

To ensure fair training, the dataset was first balanced using Random Over Sampling. This technique duplicates minority class samples until both classes have equal representation. Balancing prevents models from becoming biased toward the majority class.

### 3. Generation of Sampled Datasets

Five different probabilistic sampling techniques were applied to the balanced dataset to create five separate training subsets:

- Simple Random Sampling — selects observations randomly from the dataset.
- Systematic Sampling — selects samples at fixed intervals.
- Stratified Sampling — maintains class proportions within the sample.
- Cluster Sampling — selects groups of observations based on class clusters.
- Bootstrap Sampling — samples with replacement to create variability in the data.

Each sampling method produces a dataset with distinct statistical properties, allowing comparison of their impact on model performance.

### 4. Model Selection

Five supervised machine learning classification algorithms were selected to evaluate the effectiveness of the sampling methods:

- Logistic Regression (linear model)
- Decision Tree (rule-based model)
- Random Forest (ensemble model)
- K-Nearest Neighbors (distance-based model)
- Support Vector Machine (margin-based model)

These models represent diverse learning approaches and sensitivities to data distribution.

### 5. Train–Test Split

Each sampled dataset was divided into training and testing subsets using a 70:30 ratio. The training set was used to fit the model, while the testing set was used to evaluate predictive performance on unseen data.

### 6. Model Training

Each of the five models was trained separately on each sampled dataset. This resulted in a total of 25 experiments (5 models × 5 sampling techniques).

### 7. Performance Evaluation

Model performance was measured using classification accuracy, defined as the percentage of correctly predicted instances in the test set. Accuracy values were recorded for every model–sampling combination.

### 8. Identification of Best Sampling Technique

For each model, the sampling technique yielding the highest accuracy was identified. This allowed comparison of how different models respond to different sampling strategies.

---

## Accuracy Results

The performance of each model across different sampling techniques is shown below:

| Model        | Sampling1 | Sampling2 | Sampling3 | Sampling4 | Sampling5 |
|--------------|-----------|-----------|-----------|-----------|-----------|
| M1_Logistic  | 93.01     | 86.46     | 94.78     | 94.35     | 87.77     |
| M2_Decision  | 99.13     | 99.56     | 98.26     | 98.70     | 98.69     |
| M3_Random    | 100.00    | 100.00    | 100.00    | 100.00    | 100.00    |
| M4_KNN       | 96.51     | 95.63     | 93.91     | 93.91     | 96.51     |
| M5_SVM       | 66.81     | 69.43     | 68.70     | 67.39     | 74.67     |

---

## Best Sampling Technique per Model

| Model        | Best Sampling | Accuracy |
|--------------|---------------|----------|
| M1_Logistic  | Sampling3     | 94.78    |
| M2_Decision  | Sampling2     | 99.56    |
| M3_Random    | Sampling1     | 100.00   |
| M4_KNN       | Sampling1     | 96.51    |
| M5_SVM       | Sampling5     | 74.67    |

---

## Results Analysis

- Logistic Regression achieved its highest accuracy with Stratified Sampling, indicating that maintaining class proportions improves linear model performance.
- Decision Tree performed best with Systematic Sampling, showing robustness to structured sample selection.
- Random Forest achieved perfect accuracy across all sampling techniques, demonstrating strong generalization ability.
- KNN performed best with Simple Random Sampling, likely due to reduced noise and balanced neighborhood structure.
- SVM showed relatively lower accuracy overall but improved with Bootstrap Sampling, which increases sample diversity.

Overall, no single sampling technique was optimal for all models. Model performance depends on both the sampling method and the learning algorithm.

---

## Key Insights

- Class imbalance significantly affects model performance.
- Sampling is essential for fair training of classification models.
- Ensemble models like Random Forest are less sensitive to sampling variations.
- Distance-based models such as KNN depend heavily on data distribution.
- Bootstrap methods can improve performance for models sensitive to data variability.

---

## Tools and Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook / Google Colab

---

## Author
Name: Siya Khosla     
Roll number: 102303742

