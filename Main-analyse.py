import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Indlæs data
df = pd.read_csv("student_lifestyle_dataset.csv")

# 2. Fjern kolonner der ikke skal bruges
df = df.drop(columns=["Student_ID"])  # ID påvirker ikke stressniveau

# 3. Encoder kategorisk målvariabel (Stress_Level)
label_encoder = LabelEncoder()
df["Stress_Level"] = label_encoder.fit_transform(df["Stress_Level"])

# 4. Opdel variabler
X = df.drop(columns=["Stress_Level"])   # Features
y = df["Stress_Level"]                  # Target

# 5. Split i træning og test sæt
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Opret og træn KNN model
knn = KNeighborsClassifier(n_neighbors=5)   # du kan ændre k-værdi
knn.fit(X_train, y_train)

# 7. Forudsig
y_pred = knn.predict(X_test)

# 8. Vis performance
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Eksempel på forudsigelse:
sample = [[6.0, 3.0, 7.5, 2.0, 3.0, 3.2]]   # eksempel: indtast egen række (uden Stress_Level)
prediction = knn.predict(sample)
print("\nPredicted Stress Level:", label_encoder.inverse_transform(prediction)[0])
