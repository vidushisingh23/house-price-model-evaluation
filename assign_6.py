import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
url = 'data/train.csv'
df = pd.read_csv(url)

# Create classification target
threshold = df['SalePrice'].median()
df['PriceCategory'] = df['SalePrice'].apply(lambda x: 1 if x > threshold else 0)
df.drop(['SalePrice'], axis=1, inplace=True)

# Preprocessing
df_numeric = df.select_dtypes(include=[np.number])
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(df_numeric.drop('PriceCategory', axis=1)),
                 columns=df_numeric.drop('PriceCategory', axis=1).columns)
y = df_numeric['PriceCategory']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVC": SVC(),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

# Evaluate base models
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)

# Hyperparameter tuning
param_grid_lr = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
grid_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring='f1')
grid_lr.fit(X_train, y_train)

param_dist_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}
random_rf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist_rf,
                               n_iter=10, scoring='f1', cv=5, random_state=42)
random_rf.fit(X_train, y_train)

# Evaluate tuned models
tuned_models = {
    "Tuned Logistic Regression": grid_lr.best_estimator_,
    "Tuned Random Forest": random_rf.best_estimator_
}

for name, model in tuned_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    new_result = pd.DataFrame([{
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }])
    results_df = pd.concat([results_df, new_result], ignore_index=True)

# Plot results
plt.figure(figsize=(10, 6))
sns.barplot(x='F1-score', y='Model', data=results_df.sort_values(by='F1-score', ascending=False), palette='Blues_d')
plt.title('Model Comparison Based on F1-Score')
plt.xlabel('F1-Score')
plt.ylabel('Model')
plt.tight_layout()
plt.show()

# Final results
print("\nFinal Evaluation Results (Sorted by F1-score):")
print(results_df.sort_values(by='F1-score', ascending=False).to_string(index=False))
