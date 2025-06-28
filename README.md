
# House Price Classification with Model Evaluation & Hyperparameter Tuning

This project focuses on evaluating multiple machine learning models and applying hyperparameter tuning techniques to classify houses into **high-price** and **low-price** categories using the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) dataset.


## Problem Statement

> Given house features, classify whether the house price is **above or below the median value** using machine learning models. Evaluate each model’s performance using metrics like **accuracy**, **precision**, **recall**, and **F1-score**, and optimize them using **GridSearchCV** and **RandomizedSearchCV**.


## Dataset

- Source: `data/train.csv`
- Original Target: `SalePrice` (continuous)
- Transformed Target: `PriceCategory`  
  - 1: House price above median  
  - 0: House price below median

---

## Objectives

- Preprocess the dataset and handle missing values.
- Train multiple classification models.
- Evaluate them using standard classification metrics.
- Apply hyperparameter tuning techniques:
  - `GridSearchCV` on Logistic Regression
  - `RandomizedSearchCV` on Random Forest
- Compare performance and select the best model.

---

##  Models Used

- Logistic Regression
- Random Forest
- Support Vector Classifier (SVC)
- XGBoost Classifier
- Tuned Logistic Regression (via GridSearchCV)
- Tuned Random Forest (via RandomizedSearchCV)

---

##  Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  

All results are visualized using a **barplot** to compare model performance based on F1-score.


##  Results Snapshot

| Model                    | Accuracy | Precision | Recall | F1-score |
|--------------------------|----------|-----------|--------|----------|
| Logistic Regression      | 0.85     | 0.83      | 0.87   | 0.85     |
| Random Forest            | 0.88     | 0.86      | 0.89   | 0.87     |
| SVC                      | 0.86     | 0.85      | 0.87   | 0.86     |
| XGBoost                  | 0.89     | 0.88      | 0.89   | 0.88     |
| **Tuned Logistic Regression** | 0.86     | 0.84      | 0.88   | 0.86     |
| **Tuned Random Forest**        | **0.91** | **0.90**  | **0.92** | **0.91** |

> **Best Model**: Tuned Random Forest

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/House-Price-Classification.git
cd House-Price-Classification

# Install dependencies
pip install -r requirements.txt

# Run the Python script
python assign_6.py
```

##  Dependencies

- Python 3.10+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn


##  Learning Focus

This project demonstrates:
- Model selection through metrics comparison
- Impact of hyperparameter tuning
- Visual analysis for model evaluation
- Practical implementation of `GridSearchCV` and `RandomizedSearchCV`


##  Reference

- [KDnuggets - GridSearchCV vs. RandomizedSearchCV](https://www.kdnuggets.com/hyperparameter-tuning-gridsearchcv-and-randomizedsearchcv-explained)
- [Kaggle - House Price Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
