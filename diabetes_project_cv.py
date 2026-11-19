"""
Diabetes Risk Prediction - Term Project (Data Science + AI)

Put this file in the same folder as diabetes.csv, then run:
    python diabetes_project_cv.py

Same as the first version, plus STEP 7: 5-fold cross-validation,
which gives a more reliable comparison of the 3 models.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save charts as files instead of opening windows
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

folder = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# STEP 1: Load the data
# ------------------------------------------------------------
df = pd.read_csv(os.path.join(folder, "diabetes.csv"))
print("Rows and columns:", df.shape)
print(df.head())
print("\nOutcome counts (0 = no diabetes, 1 = diabetes):")
print(df["Outcome"].value_counts())

# ------------------------------------------------------------
# STEP 2: Clean the data
# In these columns a value of 0 is impossible, so it means "missing".
# ------------------------------------------------------------
zero_means_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_means_missing] = df[zero_means_missing].replace(0, np.nan)
print("\nMissing values per column:")
print(df.isna().sum())

# ------------------------------------------------------------
# STEP 3: Explore with charts
# ------------------------------------------------------------
# Chart 1: how many people have diabetes
df["Outcome"].value_counts().plot(kind="bar")
plt.title("Number of people with (1) and without (0) diabetes")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(folder, "chart1_outcome_counts.png"))
plt.close()

# Chart 2: glucose levels for each group
df[df["Outcome"] == 0]["Glucose"].plot(kind="hist", alpha=0.6, bins=20, label="No diabetes")
df[df["Outcome"] == 1]["Glucose"].plot(kind="hist", alpha=0.6, bins=20, label="Diabetes")
plt.title("Glucose level by outcome")
plt.xlabel("Glucose")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(folder, "chart2_glucose_by_outcome.png"))
plt.close()

# Chart 3: which features are related to the outcome
corr = df.corr()["Outcome"].drop("Outcome").sort_values()
corr.plot(kind="barh")
plt.title("Correlation of each feature with diabetes")
plt.tight_layout()
plt.savefig(os.path.join(folder, "chart3_correlation.png"))
plt.close()
print("\nSaved 3 charts as PNG files.")

# ------------------------------------------------------------
# STEP 4: Split into training data and test data
# The model learns from 80% and is tested on the 20% it has never seen.
# ------------------------------------------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------
# STEP 5: Train 3 models
# Each pipeline first fills missing values with the median,
# then (for Logistic Regression) scales the numbers.
# ------------------------------------------------------------
models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=1000)),
    "Decision Tree": make_pipeline(
        SimpleImputer(strategy="median"),
        DecisionTreeClassifier(max_depth=4, random_state=42)),
    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=200, random_state=42)),
}

# ------------------------------------------------------------
# STEP 6: Evaluate and compare
# ------------------------------------------------------------
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    })

results_df = pd.DataFrame(results).round(3)
print("\n=== MODEL COMPARISON ===")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(folder, "results.csv"), index=False)

# Which features matter most to the Random Forest?
forest = models["Random Forest"].steps[-1][1]
importance = pd.Series(forest.feature_importances_, index=X.columns).sort_values()
importance.plot(kind="barh")
plt.title("Feature importance (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(folder, "chart4_feature_importance.png"))
plt.close()
print("\nMost important features:")
print(importance.sort_values(ascending=False).round(3))

# ------------------------------------------------------------
# STEP 7: Cross-validation (more reliable comparison)
# Instead of ONE train/test split, we split the data into 5 parts.
# Each part takes a turn as the test set while the other 4 train the model.
# We then average the 5 scores. This reduces the effect of luck.
# ------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1"]

cv_rows = []
for name, model in models.items():
    # The imputer is inside the pipeline, so missing values are filled
    # using only the training part of each fold (no data leakage).
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    row = {"Model": name}
    for metric in scoring:
        values = scores["test_" + metric]
        row[metric.capitalize() + " (mean)"] = values.mean()
        row[metric.capitalize() + " (std)"] = values.std()
    cv_rows.append(row)

cv_df = pd.DataFrame(cv_rows).round(3)
print("\n=== 5-FOLD CROSS-VALIDATION (mean and spread over 5 folds) ===")
for _, r in cv_df.iterrows():
    print(f"{r['Model']:<20} "
          f"Accuracy {r['Accuracy (mean)']:.3f} +/- {r['Accuracy (std)']:.3f} | "
          f"Recall {r['Recall (mean)']:.3f} +/- {r['Recall (std)']:.3f} | "
          f"F1 {r['F1 (mean)']:.3f} +/- {r['F1 (std)']:.3f}")
cv_df.to_csv(os.path.join(folder, "cv_results.csv"), index=False)

best = cv_df.sort_values("F1 (mean)", ascending=False).iloc[0]["Model"]
print(f"\nBest model by average F1 score: {best}")

# Chart 5: accuracy and recall with error bars
x = np.arange(len(cv_df))
width = 0.35
plt.bar(x - width / 2, cv_df["Accuracy (mean)"], width,
        yerr=cv_df["Accuracy (std)"], capsize=4, label="Accuracy")
plt.bar(x + width / 2, cv_df["Recall (mean)"], width,
        yerr=cv_df["Recall (std)"], capsize=4, label="Recall")
plt.xticks(x, cv_df["Model"])
plt.ylim(0, 1)
plt.title("5-fold cross-validation results")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(folder, "chart5_cv_comparison.png"))
plt.close()
print("Saved chart5_cv_comparison.png and cv_results.csv")
