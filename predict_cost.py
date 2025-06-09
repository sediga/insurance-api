import os
import joblib
import pandas as pd

script_directory = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(script_directory, "./models/insurance_model.pkl"))

def predict_cost_risk(model, age, bmi, children, sex, smoker, region):
    # Build feature dict
    row = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0
    }
    df = pd.DataFrame([row])
    pred = model.predict(df)[0]
    return "High Cost" if pred == 1 else "Low Cost"

result = predict_cost_risk(model, age=20, bmi=33.0, children=1, sex="male", smoker="yes", region="southeast")
print("Patient Cost : "+result)
