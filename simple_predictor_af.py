# Audio Data Classification Tutorial

# This tutorial demonstrates how to prepare audio data for classification
# and how to apply machine learning models using scikit-learn.

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Step 1: Data Preparation
# ------------------------

# Read the audio features from a CSV file
af = pd.read_csv("data/audio_features/audio_features.csv")

# Extract the label from the filename in the 'AudioPath' column
af["Label"] = af["AudioPath"].apply(lambda x: x.split("/")[-1].split("_")[0])

# Convert labels from strings to integers (HC=0, PD=1)
af["Label"] = af["Label"].replace({"HC": 0, "PD": 1})

# Get ids of patients
ids = af["AudioPath"].apply(lambda x: x.split("/")[-1].split("_")[2])

# Drop the 'AudioPath' column as it's no longer needed
af.drop(columns=["AudioPath"], inplace=True)

# Split the dataset into features (X) and labels (y)
X = af.drop(columns=["Label"])
y = af["Label"]


# Split the data into training and testing sets (80% train, 20% test) ensure that ids are not in both sets
unique_ids = ids.unique()
ids_train, ids_test = train_test_split(unique_ids, test_size=0.2, random_state=42)
X_train = X[ids.isin(ids_train)]
y_train = y[ids.isin(ids_train)]
X_test = X[ids.isin(ids_test)]
y_test = y[ids.isin(ids_test)]

# Standardize the features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(
    X_train.mean(), inplace=True
)  # Use training mean to fill missing values in test set
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: RandomForest Classifier
# --------------------------------

# Initialize the RandomForestClassifier
rf = RandomForestClassifier()

# Define the parameter grid for GridSearchCV
param_grid_rf = {
    "n_estimators": [100, 300],
    "max_depth": [10, 30, 50, 70, 90, 100, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

# Perform grid search to find the best parameters
grid_search_rf = GridSearchCV(
    estimator=rf, param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=0
)
grid_search_rf.fit(X_train, y_train)

# Train the model with the best parameters found
rf_best = RandomForestClassifier(**grid_search_rf.best_params_)
rf_best.fit(X_train, y_train)

# Make predictions and evaluate the RandomForest model
y_pred_rf = rf_best.predict(X_test)
print("RandomForest Classifier Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_rf)}")
print(classification_report(y_test, y_pred_rf))

# Step 3: Logistic Regression
# ----------------------------

# Initialize the LogisticRegression
lr = LogisticRegression(max_iter=1000)

# Define the parameter grid for GridSearchCV
param_grid_lr = {
    "penalty": ["l1", "l2", "elasticnet", "none"],
    "C": [100, 10, 1.0, 0.1, 0.01],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
}

# Perform grid search to find the best parameters
grid_search_lr = GridSearchCV(
    estimator=lr, param_grid=param_grid_lr, cv=3, n_jobs=-1, verbose=0
)
grid_search_lr.fit(X_train, y_train)

# Predict with the best LogisticRegression model and evaluate
y_pred_lr = grid_search_lr.best_estimator_.predict(X_test)
print("Logistic Regression Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr)}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_lr)}")
print(classification_report(y_test, y_pred_lr))

# Conclusion
# ----------
# This tutorial demonstrated how to preprocess audio data for machine learning
# and how to apply RandomForest and LogisticRegression models to classify audio samples.
# We used GridSearchCV to optimize model parameters and evaluated our models
# using accuracy, balanced accuracy, and a detailed classification report.
