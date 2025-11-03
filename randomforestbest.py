import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# Read data
data = pd.read_csv('student_lifestyle_dataset_modifceret.csv')
data.dropna(inplace=True) 
ordenc = OrdinalEncoder(categories=[['Low', 'Moderate', 'High']])
data['Stress_Level'] = ordenc.fit_transform(data[['Stress_Level']]) + 1
#print(data.corr()['Stress_Level'].sort_values(ascending=False))

X = data.drop(columns=['Stress_Level'])
# X = data[['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Social_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 'GPA']].values
y = data['Stress_Level']

#print(data.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Create and train the model
best_model = RandomForestClassifier(n_estimators=450, max_depth=None, min_samples_split=20, min_samples_leaf=3, max_features='sqrt', bootstrap=True, random_state=42)
scores = cross_val_score(best_model, X_train, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=42))

best_model.fit(X_train, y_train)
# Evaluate the model
print(f'Cross-validation scores: {scores}')
print(f'Mean cross-validation score: {scores.mean()}')
