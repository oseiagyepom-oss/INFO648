# Decision Tree Classifier for High Earner Prediction

This notebook demonstrates the process of building and evaluating a Decision Tree Classifier to predict whether an individual is a 'high earner' based on various demographic and employment features.

## Table of Contents
1.  [Introduction to Decision Trees](#1-introduction-to-decision-trees)
2.  [Data Loading and Initial Exploration](#2-data-loading-and-initial-exploration)
3.  [Defining the Target Variable: `high_earner`](#3-defining-the-target-variable-high_earner)
4.  [Train-Test Split](#4-train-test-split)
5.  [Feature Engineering and Preprocessing](#5-feature-engineering-and-preprocessing)
6.  [Building and Training the Decision Tree Model](#6-building-and-training-the-decision-tree-model)
7.  [Model Evaluation](#7-model-evaluation)
8.  [Prediction on New Data](#8-prediction-on-new-data)

## 1. Introduction to Decision Trees
This section provides an overview of Decision Trees, their components (root nodes, branches, leaf nodes), and why they are a popular choice for classification and regression tasks due to their interpretability and ability to handle various data types.

## 2. Data Loading and Initial Exploration
The notebook starts by loading the `cps_with_education_clean.csv` dataset into a pandas DataFrame. Basic data exploration, including `df.head()` and `df.describe()`, is performed to understand the dataset structure and summary statistics.

```python
import pandas as pd
df = pd.read_csv('/content/cps_with_education_clean.csv')
df.head()
df.describe()
3. Defining the Target Variable: high_earner

A new binary target variable, high_earner, is created. Individuals earning above the median weekly earnings (EARNWEEK2) are labeled as 1, and those at or below the median are labeled as 0.

high_pay = model_df['EARNWEEK2'].median()
model_df['high_earner'] = (model_df['EARNWEEK2'] > high_pay).astype(int)
model_df['high_earner'].value_counts()
4. Train-Test Split

The data is split into training and testing sets using train_test_split to ensure robust model evaluation. Features include 'AGE', 'UHRSWORKT', 'education', and 'SEX', with 'high_earner' as the target.

from sklearn.model_selection import train_test_split
X = model_df[["AGE","UHRSWORKT","education","SEX"]]
y = model_df["high_earner"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
5. Feature Engineering and Preprocessing

Both numeric (AGE, UHRSWORKT) and categorical (education, SEX) features are identified. A ColumnTransformer is used to apply one-hot encoding to categorical features. Scaling is noted as not strictly necessary for Decision Trees but is good practice for other models.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

numeric_features = ["AGE", "UHRSWORKT"]
categorical_features = ["education", "SEX"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder='passthrough'
)
6. Building and Training the Decision Tree Model

A Pipeline is constructed to combine the preprocessing step with a DecisionTreeClassifier. The model is then trained on the X_train and y_train data.

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

model = Pipeline([
    ("prep", preprocess),
    ("clf", DecisionTreeClassifier(max_depth=10, min_samples_leaf=20))
])

model.fit(X_train, y_train)
The trained Decision Tree can also be visualized to understand its decision rules.

from sklearn.tree import export_graphviz
import graphviz

tree = model.named_steps["clf"]
feature_names = model.named_steps["prep"].get_feature_names_out()

dot = export_graphviz(
    tree,
    out_file=None,
    feature_names=feature_names,
    class_names=["Low", "High"],
    filled=True,
    rounded=True,
    special_characters=True,
    max_depth=4
)

graphviz.Source(dot)
7. Model Evaluation

The model's performance is evaluated using various metrics:

ROC AUC Score: Calculated using roc_auc_score to assess the model's ability to distinguish between classes.
ROC Curve: Visualized to show the trade-off between True Positive Rate and False Positive Rate at different thresholds.
Confusion Matrix: Provides a breakdown of true positives, true negatives, false positives, and false negatives.
Classification Report: Presents precision, recall, f1-score, and support for each class.
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay

y_prob = model.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_prob)

RocCurveDisplay.from_estimator(model, X_test, y_test, plot_chance_level=True)

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
Threshold adjustment is also explored to see its impact on the classification results.

y_pred_Num = (y_prob >= 0.48).astype(int)
print(confusion_matrix(y_test, y_pred_Num))
print(classification_report(y_test, y_pred_Num, digits=3))
8. Prediction on New Data

Finally, the trained model is used to make predictions on a new, unseen dataset (new_donors.csv), demonstrating how to apply the pipeline to new data.

new_data = pd.read_csv('/content/new_donors.csv')
print(new_data)
model.predict(new_data)
model.predict_proba(new_data)
This notebook provides a comprehensive guide to building, training, and evaluating a Decision Tree Classifier for a binary classification problem.


