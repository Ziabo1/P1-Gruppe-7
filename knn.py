import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# --- Data Preparation ---
def DataPrep():
    data = pd.read_csv('student_lifestyle_dataset_modificeret.csv')
    data.dropna(inplace=True)
    ordenc = OrdinalEncoder(categories=[['Low', 'Moderate', 'High']])
    data['Stress_Level'] = (ordenc.fit_transform(data[['Stress_Level']]) + 1).astype(int)
    X = data.drop(columns=['Stress_Level'])
    y = data['Stress_Level']
    return X, y

# --- Train/Test Split ---
def Split():
    X, y = DataPrep()
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- KNN Optimization ---
def KNNOptim(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    param_grid = {
        'knn__n_neighbors': list(range(1, 31)),
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    X_train, X_test, y_train, y_test = Split()
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# --- Return Best Model ---
def KNNBest(X_train, y_train):
    knn = KNNOptim(X_train, y_train)
    return knn

# --- Validate Model ---
def Validate():
    X_train, X_test, y_train, y_test = Split()
    best_model = KNNBest(X_train, y_train)
    scores = cross_val_score(best_model, X_train, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=42))
    print(f'Cross-validation scores(KNN): {scores}')
    print(f'Mean cross-validation score(KNN): {scores.mean():.4f}')

