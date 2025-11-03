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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

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
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_grid = {
    'knn__n_neighbors': list(range(1, 31)),
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]   # 1->Manhattan, 2->Euclidean
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)
print("best params (kNN):", grid.best_params_)
print("best CV score (kNN):", grid.best_score_)
best_knn = grid.best_estimator_
scores = cross_val_score(best_knn, X_train, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=42))
best_knn.fit(X_train, y_train)
# Evaluate the model
print(f'Cross-validation scores: {scores}')
print(f'Mean cross-validation score: {scores.mean()}')
