

# import packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import imblearn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import csv

def Test_performance(approach,model,X_test,y_test,performance_df,performance_file,classwise_precision_file, classwise_recall_file,classwise_f1_file):
    print(approach)
    y_pred = model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    class_report = classification_report(y_test, y_pred)
    print("Classification Report:\n", class_report)
    # Collect metrics
    row = {
        "approach": approach,                             # <-- fill in your model name
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average='macro'),
        "recall_macro": recall_score(y_test, y_pred, average='macro'),
        "f1_macro": f1_score(y_test, y_pred, average='macro'),
        "precision_weighted": precision_score(y_test, y_pred, average='weighted'),
        "recall_weighted": recall_score(y_test, y_pred, average='weighted'),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted')
    }

    # Append row
    performance_df = pd.concat([performance_df, pd.DataFrame([row])], ignore_index=True)

    with open(performance_file, "a", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=["approach", "accuracy", "precision_macro", "recall_macro", "f1_macro","precision_weighted", "recall_weighted", "f1_weighted"])

      writer.writerow(row)

    # ------------------------------
    # NEW: Classwise Precision
    # ------------------------------
    precision_row = {"approach": approach}
    for cls in ["0", "1", "2", "3", "4"]:
        precision_row[f"Stage {cls}"] = report_dict[cls]["precision"]

    with open(classwise_precision_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=precision_row.keys())
        writer.writerow(precision_row)
    

    # ------------------------------
    # NEW: Classwise Recall
    # ------------------------------
    recall_row = {"approach": approach}
    for cls in ["0", "1", "2", "3", "4"]:
        recall_row[f"Stage {cls}"] = report_dict[cls]["recall"]

    with open(classwise_recall_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=recall_row.keys())
        writer.writerow(recall_row)

    # ------------------------------
    # NEW: Classwise F1-score
    # ------------------------------
    f1_row = {"approach": approach}
    for cls in ["0", "1", "2", "3", "4"]:
        f1_row[f"Stage {cls}"] = report_dict[cls]["f1-score"]

    with open(classwise_f1_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=f1_row.keys())
        writer.writerow(f1_row)


def KFold_Cross_Validation(model,X,y,k=5):
    print(f"K-Fold Cross Validation with k={k}")
    kf = KFold(n_splits=k, shuffle=True, random_state=33)
    fold = 1
    for train_index, test_index in kf.split(X):
        print(f"Fold {fold}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        class_report = classification_report(y_test, y_pred)
        print("Classification Report:\n", class_report)
        fold += 1
    

# Change this with the path to the "oral_cancer_prediction_dataset.csv"
base_dir = ""
path = base_dir + "oral_cancer_prediction_dataset.csv"

print("Path to dataset files:", path)

df=pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(f"Have nulls: {df.isnull().values.any()}")

"""# Pre-processing the data"""

# encode the labels
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# list of columns to drop
cols_to_drop = [
    'ID',
    'Country',
    'Treatment Type',
    'Survival Rate (5-Year, %)',
    'Cost of Treatment (USD)',
    'Economic Burden (Lost Workdays per Year)',
    'Early Diagnosis',
    'Oral Cancer (Diagnosis)',
    'Diet (Fruits & Vegetables Intake)'
]

target = 'Cancer Stage'
y = df[target]
X = df.drop(columns=cols_to_drop + [target], errors='ignore')

# store X and y in csv files
X.to_csv(base_dir+"X.csv",index=False)
y.to_csv(base_dir+"y.csv",index=False)

# This part checks the comparative ratio of the labels
y_sorted=sorted(y)

# count number of each label

label_counts = {}
for label in y_sorted:
    if label in label_counts:
        label_counts[label] += 1
    else:
        label_counts[label] =1
print("Count of the labels:",label_counts)
# normalize the count

total_count = sum(label_counts.values())
normalized_counts = {label: count / total_count*100 for label, count in label_counts.items()}
for label, normalized_count in normalized_counts.items():
    print(f"{label}: {normalized_count:.2f}%")

# a figure with the label ratio

labels = list(normalized_counts.keys())
values = list(normalized_counts.values())

plt.figure(figsize=(8, 6))
plt.bar(labels, values, color='skyblue')
plt.xlabel('Labels')
plt.ylabel('Ratio(%)')
plt.title('Label Ratio')


plt.savefig(base_dir+"label_ratio.png")

# splitting train and test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,       # 30% test
    random_state=3,      # for reproducibility
    stratify=y            #keeps class proportions the same
)

numeric_features=['Age','Tumor Size (cm)']
boolean_features=X_train.columns.difference(numeric_features).tolist()
categorical_features=[target]

print("Numeric features ")
print(numeric_features)
print("Boolean features ")
print(boolean_features)

# Do normalization and generate oversampled dataset to make a balanced set of points
scaler = StandardScaler()
# just sclale numeric features

X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])
# oversampling using SMOTE
sm = SMOTE(random_state=33)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# create a csv file to store the performace for different approaches. rows will be approach and columns will be different validation parameters
performance_file=base_dir+"performance.csv"
# declare header for the csv file


print("Performance file path:", performance_file )
header = ["Approach", "accuracy", "precision_macro", "recall_macro", "f1_macro","precision_weighted", "recall_weighted", "f1_weighted"]

# Create empty DF with header
performance_df = pd.DataFrame(columns=header)

performance_df.to_csv(performance_file, index=False)

classwise_precision_file = base_dir + "precision_classwise.csv"
classwise_recall_file = base_dir + "recall_classwise.csv"
classwise_f1_file = base_dir + "f1_classwise.csv"

# Create empty files with header
pd.DataFrame(columns=["approach", "class_0", "class_1", "class_2", "class_3", "class_4"]).to_csv(classwise_precision_file, index=False)
pd.DataFrame(columns=["approach", "class_0", "class_1", "class_2", "class_3", "class_4"]).to_csv(classwise_recall_file, index=False)
pd.DataFrame(columns=["approach", "class_0", "class_1", "class_2", "class_3", "class_4"]).to_csv(classwise_f1_file, index=False)


"""# Logistic Regression"""


model = LogisticRegression(class_weight='balanced', penalty="l2", multi_class='multinomial', max_iter=1000)
model.fit(X_train, y_train)
Test_performance("Logistic Regression",model,X_test,y_test,performance_df,performance_file,classwise_precision_file,classwise_recall_file,classwise_f1_file)
#KFold_Cross_Validation(model,X,y,k=5)


"""The scores indicate that the model overfits on the label "0" data points. Only a very few of the test data points from other labels are correctly detected.

Try with oversampled data
"""

model = LogisticRegression(class_weight='balanced', penalty="l2", multi_class='multinomial', max_iter=1000)
model.fit(X_train_res, y_train_res)
Test_performance("Logistic Regression With Oversampling",model,X_test,y_test,performance_df,performance_file,classwise_precision_file,classwise_recall_file,classwise_f1_file)
#KFold_Cross_Validation(model,X,y,k=5)

"""No significant performance improvement is seen after oversampling. Seems like the unbalanced labels is not the problem here. We will try with other models.

# Naive Bayes
"""

model = GaussianNB()
model.fit(X_train, y_train)
Test_performance("Naive Bayes",model,X_test,y_test,performance_df,performance_file,classwise_precision_file,classwise_recall_file,classwise_f1_file)
#KFold_Cross_Validation(model,X,y,k=5)

"""On oversampled data"""

model = GaussianNB()
model.fit(X_train_res, y_train_res)
Test_performance("Naive Bayes With Oversampling",model,X_test,y_test,performance_df,performance_file,classwise_precision_file,classwise_recall_file,classwise_f1_file)
#KFold_Cross_Validation(model,X,y,k=5)

"""# Random Forest"""

model = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=1000,
    max_depth=None,
    random_state=33
)
model.fit(X_train, y_train)
Test_performance("Random Forest",model,X_test,y_test,performance_df,performance_file,classwise_precision_file,classwise_recall_file,classwise_f1_file)
#KFold_Cross_Validation(model,X,y,k=5)

"""With SMOTE oversampling"""

model = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=1000,
    max_depth=None,
    random_state=33
)
model.fit(X_train_res, y_train_res)
Test_performance("Random Forest With Oversampling",model,X_test,y_test,performance_df,performance_file,classwise_precision_file,classwise_recall_file,classwise_f1_file)
#KFold_Cross_Validation(model,X,y,k=5)

"""# XGBoost"""

model = XGBClassifier(
    objective="multi:softmax",
    num_class=5,
    eval_metric="mlogloss",
    n_estimators=400,
    learning_rate=0.08,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist"
)
model.fit(X_train, y_train)
Test_performance("XGBoost",model,X_test,y_test,performance_df,performance_file,classwise_precision_file,classwise_recall_file,classwise_f1_file)
#KFold_Cross_Validation(model,X,y,k=5)

"""# On oversampled data"""

model = XGBClassifier(
    objective="multi:softmax",
    num_class=5,
    eval_metric="mlogloss",
    n_estimators=400,
    learning_rate=0.08,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist"
)
model.fit(X_train_res, y_train_res)
Test_performance("XGBoost With Oversampling",model,X_test,y_test,performance_df,performance_file,classwise_precision_file,classwise_recall_file,classwise_f1_file)

"""# Neural Networks"""

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),   # 2 hidden layers
    activation='relu',
    solver='adam',
    alpha=0.0005,                    # L2 regularization
    max_iter=500,
    learning_rate='adaptive',
    random_state=33
)

model.fit(X_train, y_train)

Test_performance("Neural Network",model,X_test,y_test,performance_df,performance_file,classwise_precision_file,classwise_recall_file,classwise_f1_file)

"""On oversampled data"""

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),   # 2 hidden layers
    activation='relu',
    solver='adam',
    alpha=0.0005,                    # L2 regularization
    max_iter=500,
    learning_rate='adaptive',
    random_state=33
)

model.fit(X_train_res, y_train_res)

Test_performance("Neural Network With Oversampling",model,X_test,y_test,performance_df,performance_file,classwise_precision_file,classwise_recall_file,classwise_f1_file)


# run

# python3 cse5819_project.py
