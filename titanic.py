# Titanic Survival Project - Mrinalika Ampagowni

# 1.1 Data Cleansing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train_df = pd.read_csv("data/train.csv")

train_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

train_df["Age"].fillna(train_df["Age"].median(), inplace=True)
train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)

#1.2 Data Analysis
sns.countplot(data=train_df, x="Survived", hue="Sex")
plt.title("Survival Count by Sex")
plt.show()

sns.countplot(data=train_df, x="Survived", hue="Pclass")
plt.title("Survival Count by Pclass")
plt.show()

train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=[0, 12, 20, 40, 60, 80], labels=['Child', 'Teen', 'Adult', 'Mid-Age', 'Senior'])
sns.countplot(data=train_df, x='AgeGroup', hue='Survived')
plt.title("Survival Count by Age Group")
plt.show()
train_df.drop(columns=["AgeGroup"], inplace=True)

#1.3 Preprocessing
le = LabelEncoder()
train_df["Sex"] = le.fit_transform(train_df["Sex"])
train_df["Embarked"] = le.fit_transform(train_df["Embarked"])

X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#1.4 Modeling
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train, y_train)

#1.5 Tuning
param_grid = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_tree = grid_search.best_estimator_

#1.6 Evaluate
models = {"Logistic Regression": logreg, "Decision Tree": best_tree}
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

#1.7 Report
print("\nFinal Report Summary:")
print(f"Best Decision Tree Parameters: {grid_search.best_params_}")
