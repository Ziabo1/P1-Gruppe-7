import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold

#Indlæs datasættet
#Info på data og kontrol af felter med null
student_dataset = pd.read_csv("student_lifestyle_dataset_modificeret.csv")
print("\n--- INFO ---")
print(student_dataset.info())

print("\n--- DESCRIPTION (NUMERIC) ---")
print(student_dataset.describe())

print("\n--- FIRST ROWS ---")
print(student_dataset.head())

print("\n--- MISSING VALUES ---")
print(student_dataset.isnull().sum())


# Undersøgelse af data
# skelne mellem numerisk og kategorisk data
numeric_cols = student_dataset.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = student_dataset.select_dtypes(include=['object']).columns

# Histogrammer for numerisk data
student_dataset[numeric_cols].hist(figsize=(12, 8), bins=20)
plt.suptitle("Feature Distributions (Numeric)", fontsize=14)
plt.show()


# Countplots for kategoriske features
# Countplot for Stress Level in correct order
plt.figure(figsize=(6,4))
sns.countplot(
    data=student_dataset,
    x="Stress_Level",
    order=["Low", "Moderate", "High"]
)
plt.title("Countplot - Stress Level")
plt.xticks(rotation=30)
plt.show()


# Boxplots to check for outliers (numeric)
# for col in numeric_cols:
#     plt.figure(figsize=(6,4))
#     sns.boxplot(data=student_dataset, y=col)
#     plt.title(f"Boxplot - {col}")
#     plt.show()


# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(student_dataset[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


#ordinal encode stress niveau
ord_encoder = OrdinalEncoder(categories=[["Low", "Moderate", "High"]])
student_dataset["Stress_Level"] = ord_encoder.fit_transform(student_dataset[["Stress_Level"]]) + 1

print(student_dataset.head())

# Opdel data i features (X) og target (y)
X = student_dataset.drop(columns=["Stress_Level"])
y = student_dataset["Stress_Level"]

# Train test split, 80/20.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Standard scaling af numerisk data
std_scaler = StandardScaler()
X_train_std_scal = std_scaler.fit_transform(X_train)
X_test_std_scal = std_scaler.transform(X_test)


# Convert scaled train data to DataFrame
X_train_scaled_df = pd.DataFrame(X_train_std_scal, columns=X.columns)

# Pairplot (good for cluster overview)
# --- BEFORE vs AFTER STANDARD SCALER VISUALISATION ---

# Convert original and scaled data to DataFrames for plotting
X_train_before_df = pd.DataFrame(X_train, columns=X.columns)
X_train_after_df = pd.DataFrame(X_train_std_scal, columns=X.columns)

# Plot KDE before and after scaling
plt.figure(figsize=(12,5))

# BEFORE
plt.subplot(1, 2, 1)
for col in X_train_before_df.columns:
    sns.kdeplot(X_train_before_df[col], label=col, fill=True)
plt.title("Before Scaling")
plt.xlabel("Feature Value")
plt.ylabel("Density")
plt.legend()

# AFTER
plt.subplot(1, 2, 2)
for col in X_train_after_df.columns:
    sns.kdeplot(X_train_after_df[col], label=col, fill=True)
plt.title("After Standard Scaler")
plt.xlabel("Scaled Feature Value")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()
plt.show()



# Træning af model
# Grid search
# -------------------------
# KNN Model Training
# -------------------------

# Initialize KNN
knn = KNeighborsClassifier()

# Use cross-validation inside the training set
param_grid = {'n_neighbors': list(range(3, 16)), 'weights': ['uniform', 'distance']}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train_std_scal, y_train)

print("Best KNN Parameters:", grid_search.best_params_)
best_knn = grid_search.best_estimator_

# Cross-validation results on training set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(best_knn, X_train_std_scal, y_train, cv=cv, scoring='accuracy')
cv_f1 = cross_val_score(best_knn, X_train_std_scal, y_train, cv=cv, scoring='f1_macro')

print("\n--- CROSS VALIDATION RESULTS ---")
print(f"Mean Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"Mean F1 (Macro): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")


#Træne Random Forest model

