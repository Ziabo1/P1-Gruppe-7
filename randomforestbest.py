import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# Read data
def DataPrep():
    data = pd.read_csv('student_lifestyle_dataset_modifceret.csv')
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
        'n_estimators': [100, 150, 175, 200, 250, 275, 300],
        'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'min_samples_split': [2, 5, 10, 15, 20, 25, 30, 35, 40],
        'min_samples_leaf': [1,  2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'max_features': [None, 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param, n_iter=100, cv=3, verbose=0, random_state=42, n_jobs=-1, scoring='accuracy')
    random_search.fit(X_train, y_train)
    #print(f'Best parameters found: {random_search.best_params_}')
    #print(f'Best RandomSearch score: {random_search.best_score_:.4f}')
    bestparams = random_search.best_params_

    paramgrid = {
        'n_estimators': [bestparams['n_estimators'] - 50, bestparams['n_estimators'], bestparams['n_estimators'] + 50],
        'max_depth': [bestparams['max_depth'] - 5 if bestparams['max_depth'] is not None else None, bestparams['max_depth'], bestparams['max_depth'] + 5 if bestparams['max_depth'] is not None else None],
        'min_samples_split': [bestparams['min_samples_split'] - 1, bestparams['min_samples_split'], bestparams['min_samples_split'] + 1],
        'min_samples_leaf': [bestparams['min_samples_leaf'] - 1, bestparams['min_samples_leaf'], bestparams['min_samples_leaf'] + 1],
        'max_features': [bestparams['max_features']],
        'bootstrap': [bestparams['bootstrap']]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=paramgrid, cv=5, n_jobs=-1, scoring='accuracy', verbose=0)
    grid_search.fit(X_train, y_train)
    #print(f'Best parameters found: {grid_search.bestparams}')
    #print(f'Best GridSearch score: {grid_search.bestscore:.4f}')
    
    return grid_search.best_estimator_
def RFBest(X_train, y_train):
    rf = RFOptim(X_train, y_train)
    return rf
def Validate():
    X_train, X_test, y_train, y_test = Split()
    best_model = RFBest(X_train, y_train)
    scores = cross_val_score(best_model, X_train, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=42), scoring='accuracy')
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
def Test():
    X_train, X_test, y_train, y_test = Split()
    finalmodel = RFBest(X_train, y_train)
    finalmodel.fit(X_train, y_train)
    y_pred = finalmodel.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    print("Test F1 score:", f1score)
    print("Test accuracy:", accuracy)
    print("Test precision:", precision)
    print("Test recall:", recall)
def TrainAndSaveModel():
    X_train, X_test, y_train, y_test = Split()
    model = RFBest(X_train, y_train)
    model.fit(X_train, y_train)
    print("Saving model to:", os.getcwd())
    joblib.dump(model, "stress_model.pkl")
    joblib.dump(X_train.columns.tolist(), "feature_names.pkl")
# Evaluate the model