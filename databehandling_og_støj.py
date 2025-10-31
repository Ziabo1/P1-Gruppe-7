import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random

#Indlæs datasættet
#Info på data og kontrol af felter med null
student_dataset = pd.read_csv("student_lifestyle_dataset.csv")
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

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(student_dataset[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#dropna på student ID
student_dataset = student_dataset.drop(columns=["Student_ID"])



# Scatterplot over study hours og sleep hours med stress level plottet.

# Example numeric columns to plot — replace these with the ones you want
x_col = "Sleep_Hours_Per_Day"
y_col = "Study_Hours_Per_Day"

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=student_dataset,
    x=x_col,
    y=y_col,
    hue="Stress_Level",              # Color by stress level
    palette={"Low": "green", "Moderate": "orange", "High": "red"},
    s=80,
    edgecolor="black"
)

# Optional: Add stress level text on each datapoint
for i in range(len(student_dataset)):
    plt.text(
        student_dataset[x_col].iloc[i],
        student_dataset[y_col].iloc[i],
        student_dataset["Stress_Level"].iloc[i][0],  # show first letter (L/M/H)
        fontsize=8,
        ha='center',
        va='center'
    )

plt.title(f"Scatterplot of study hours per day vs sleep hours per day (Stress Level shown on each point)")
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.legend(title="Stress Level")
plt.grid(True)
plt.tight_layout()
plt.show()


# Code for creating noise on the data.
# Add ~5–10% random Gaussian noise
noise_fraction = 0.1
for col in ["Study_Hours_Per_Day", "Sleep_Hours_Per_Day", "Social_Hours_Per_Day", "GPA"]:
    std = student_dataset[col].std()
    noise = np.random.normal(0, noise_fraction * std, size=len(student_dataset))
    student_dataset[col] = (student_dataset[col] + noise).clip(lower=0)  # no negative hours



flip_fraction = 0.05  # 5% of samples
n = len(student_dataset)
flip_indices = random.sample(range(n), int(flip_fraction * n))
classes = student_dataset['Stress_Level'].unique()

for i in flip_indices:
    current = student_dataset.loc[i, 'Stress_Level']
    student_dataset.loc[i, 'Stress_Level'] = random.choice([c for c in classes if c != current])





# Pick the same numeric features as before
x_col = "Sleep_Hours_Per_Day"
y_col = "Study_Hours_Per_Day"

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=student_dataset,
    x=x_col,
    y=y_col,
    hue="Stress_Level",
    palette={"Low": "green", "Moderate": "orange", "High": "red"},
    s=80,
    edgecolor="black",
    alpha=0.8
)

# Optional: label each point with its (possibly flipped) stress level initial
for i in range(len(student_dataset)):
    plt.text(
        student_dataset[x_col].iloc[i],
        student_dataset[y_col].iloc[i],
        student_dataset["Stress_Level"].iloc[i][0],  # L/M/H
        fontsize=8,
        ha='center',
        va='center'
    )

plt.title(f"Scatterplot After Noise & Label Flip\n(study hours per day vs sleep hours per day with Stress Level)")
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.legend(title="Stress Level")
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the modified dataset to a new CSV file
student_dataset.to_csv("student_lifestyle_dataset_modifceret.csv", index=False)

print("\n--- New dataset saved as 'student_lifestyle_dataset_modified.csv' ---")