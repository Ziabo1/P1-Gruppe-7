import pandas as pd
import joblib
import os

def get_user_input(feature_names):
    print("Du bliver nu bedt om at uddele dit døgn, altså 24 timer, ud i 5 forskellige kategorier. " \
    "Derudover bliver du spurgt om din GPA, altså karaktergennemsnit, som skal ligge mellem 0 0g 4. " \
    "Indtast venligst dine data:")

    data = {}
    timer = 0
    for feature in feature_names:
        value = float(input(f"{feature}:"))
        if feature != "GPA":
            timer += value
        data[feature] = value

    if timer != 24:
        print("Fejl: Summen af timer skal være 24. Prøv igen.")
        return get_user_input(feature_names)

    return pd.DataFrame([data])

def predict_stress():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(script_dir, "stress_model.pkl"))
    feature_names = joblib.load(os.path.join(script_dir, "feature_names.pkl"))
    
    user_df = get_user_input(feature_names)
    prediction = model.predict(user_df)[0]

    stress_map = {
        1: "Low",
        2: "Moderate",
        3: "High"
    }

    print("\nForudsagt stressniveau:", stress_map[prediction])

if __name__ == "__main__":
    predict_stress()