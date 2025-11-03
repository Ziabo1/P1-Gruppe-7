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
model = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [50, 100, 150, 200, 300, 500, 1000, 1500, 2000],
    'max_depth': [None, 10, 20, 30, 50, 70, 100, 150, 200, 300, 500],
    'min_samples_split': [2, 5, 10, 15, 20, 25, 30, 50, 100],
    'min_samples_leaf': [1, 2, 4, 8, 10, 15, 20, 25, 30, 50],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=5, random_state=42)
random_search.fit(X_train, y_train)

print(random_search.best_params_)
print(random_search.best_score_)

param_dis = {
    'n_estimators': [450, 500, 550],
    'max_depth': [None, 10, 20],
    'min_samples_split': [20, 25, 30],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_dis,
    cv=5,                 # 5-fold cross-validation
    scoring='accuracy',   # metrik du optimerer på
    n_jobs=-1,            # brug alle CPU-kerner
    verbose=2             # viser fremgang i konsollen
)
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)
scores = cross_val_score(model, X_train, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=42))
scores2 = cross_val_score(best_model, X_train, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=42))

#model.fit(X_train, y_train)
# Evaluate the model
#print(f'Cross-validation scores: {scores}')
#print(f'Mean cross-validation score: {scores.mean()}')
#print(f'Cross-validation scores (best model): {scores2}')
#print(f'Mean cross-validation score (best model): {scores2.mean()}')

