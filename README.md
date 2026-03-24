# House Price Prediction — ML Regression Model

## Overview
An end-to-end machine learning pipeline to predict California house prices using regression models.

## Dataset
- California Housing Dataset (scikit-learn built-in)
- 19,607 records after cleaning

## Tech Stack
- Python, Scikit-Learn, Pandas, Matplotlib, Seaborn

## Steps
1. Exploratory Data Analysis (EDA)
2. Outlier treatment and data cleaning
3. Feature engineering (rooms_per_person, bedrooms_ratio, income_rooms)
4. Model training — Linear Regression, Ridge, Random Forest, Gradient Boosting
5. Cross-validation and GridSearchCV tuning
6. Feature importance and residual visualisation

## Results
| Model | R² Score |
|-------|----------|
| Linear Regression | 0.6248 |
| Ridge | 0.6248 |
| Gradient Boosting | 0.7576 |
| **Random Forest** | **0.7768** ✅ |

## Key Findings
- Median Income is the strongest predictor of house price (importance ~0.45)
- Location (Latitude/Longitude) is the second most important factor
- Tuned Random Forest achieves MAE of $31,320 on test data