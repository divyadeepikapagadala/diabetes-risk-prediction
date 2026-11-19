# Diabetes Risk Prediction Using Machine Learning

By Gnana Deepika (Roll Number: 23035010864)

Term project (DA 377), B.Sc. (Hons.) Data Science and AI, IIT Guwahati.

This project predicts whether a person has diabetes from eight basic health measurements, and compares three models: Logistic Regression, Decision Tree and Random Forest, using an 80/20 split and 5-fold cross-validation.

## Dataset
Pima Indians Diabetes Database (768 records), downloaded from Kaggle:
https://www.kaggle.com/datasets/mragpavank/diabetes

Download `diabetes.csv` and put it in the same folder as the scripts.

## How to run
pip install pandas scikit-learn matplotlib
python diabetes_project_cv.py

## Files
- `diabetes_project.py`: cleaning, charts, three models, single-split results
- `diabetes_project_cv.py`: the same, plus 5-fold cross-validation

## Note
This is a learning project, not a medical tool. The best models still miss about four in ten people with diabetes.
