# ML Model Analysis on Student Performance & Air Quality Dataset

## Project Overview

This project analyzes how environmental and cognitive factors affect student performance using machine learning models.

The project compares multiple classification algorithms to evaluate their ability to predict student performance levels based on:
- air quality measurements
- cognitive indicators
- educational metrics
- environmental conditions

## Objectives

The main objectives of this project are:

- Build a complete machine learning pipeline
- Compare multiple classification algorithms
- Analyze model generalization and trade-offs
- Apply preprocessing and feature engineering techniques
- Visualize high-dimensional data using PCA
- Evaluate model performance using multiple metrics

## Dataset 

The dataset was taken from https://www.kaggle.com/datasets/uniquetech/air-quality-and-student-performance-dataset and contains student learning, cognitive, and environmental information including:

- CO2 concentration
- PM2.5 levels
- temperature
- humidity
- focus rating
- reaction time
- quiz score
- subject information
- air quality labels

### Target Variable
`performance_label`

Classes:
- High
- Medium
- Low

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Preprocessing

Several preprocessing techniques were applied:

### 1. ID Column Removal
The `student_id` column was removed because it does not contain meaningful predictive information.

### 2. Encoding Strategies

#### Ordinal Encoding
Used for:
- `grade`
- `air_quality_label`

These variables contain meaningful ranking relationships.

#### One-Hot Encoding
Used for:
- `day`
- `period`
- `subject`

These variables do not contain meaningful numerical ordering.

### 3. Feature Scaling
StandardScaler was used to normalize feature ranges for:
- Logistic Regression
- KNN
- PCA

## Models Used

### Logistic Regression
Used as a strong interpretable baseline classifier.

### K-Nearest Neighbors (KNN)
Used to analyze distance-based learning behavior.

### Random Forest
Used to capture nonlinear relationships and improve generalization through ensemble learning.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Weighted averaging was used because the dataset contains multiple classes.

