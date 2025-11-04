import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# Read data
def DataPrep():
    data = pd.read_csv('student_lifestyle_dataset_modificeret.csv')
    data.dropna(inplace=True) 
    ordenc = OrdinalEncoder(categories=[['Low', 'Moderate', 'High']])
    data['Stress_Level'] = (ordenc.fit_transform(data[['Stress_Level']]) + 1).astype(int)
    X = data.drop(columns=['Stress_Level'])
    y = data['Stress_Level']
    return X, y

#print(data.head())
def Split():
    X, y = DataPrep()
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Create and train the model
def RFOptim(X_train, y_train):
    param = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param, n_iter=100, cv=3, verbose=0, random_state=42, n_jobs=-1, scoring='f1_macro')
    random_search.fit(X_train, y_train)
    print(f'Best parameters found: {random_search.best_params_}')
    print(f'Best RandomSearch score: {random_search.best_score_:.4f}')
    return random_search.best_estimator_
def RFBest(X_train, y_train):
    rf = RFOptim(X_train, y_train)
    return rf
def Validate():
    X_train, X_test, y_train, y_test = Split()
    best_model = RFBest(X_train, y_train)
    scores = cross_val_score(best_model, X_train, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=42), scoring='f1_macro')
    importance = best_model.feature_importances_
    print(f'Cross-validation scores(RF): {scores}')
    print(f'Mean cross-validation score(RF): {scores.mean():.4f}')
    feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance
    }).sort_values('Importance', ascending=False)
    plt.figure(figsize=(8, 5))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.gca().invert_yaxis()  # highest importance at top
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.savefig('random_forest_feature_importance.png')
# Evaluate the model