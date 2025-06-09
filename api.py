from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

# Load model
model_path = os.path.join(os.path.dirname(__file__), "models", "insurance_model.pkl")
model = joblib.load(model_path)

# Input schema
class InsuranceInput(BaseModel):
    age: int
    bmi: float
    children: int
    sex: str         # "male" or "female"
    smoker: str      # "yes" or "no"
    region: str      # "southeast", "southwest", "northwest", "northeast"

@app.post("/predict")
def predict(input: InsuranceInput):
    row = {
        "age": input.age,
        "bmi": input.bmi,
        "children": input.children,
        "sex_male": 1 if input.sex == "male" else 0,
        "smoker_yes": 1 if input.smoker == "yes" else 0,
        "region_northwest": 1 if input.region == "northwest" else 0,
        "region_southeast": 1 if input.region == "southeast" else 0,
        "region_southwest": 1 if input.region == "southwest" else 0
    }
    df = pd.DataFrame([row])
    prediction = model.predict(df)[0]
    return {"prediction": "High Cost 💸" if prediction == 1 else "Low Cost ✅"}
