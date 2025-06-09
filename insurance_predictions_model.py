# Step 1: Import libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

script_directory = os.path.dirname(os.path.abspath(__file__))

# Step 2: Load dataset
df = pd.read_csv(os.path.join(script_directory, "../data/insurance.csv"))

# Step 3: Create binary label (1 = High Cost, 0 = Low Cost)
df["high_cost"] = (df["charges"] > df["charges"].median()).astype(int)

# Step 4: One-hot encode categorical columns
df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

# Step 5: Define features and target
X = df.drop(["charges", "high_cost"], axis=1)
y = df["high_cost"]

# Step 6: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the model
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, os.path.join(script_directory, "./models/insurance_model.pkl"))

# Step 10: Feature Importance Plot
# importances = model.feature_importances_
# features = X.columns

# plt.figure(figsize=(10, 6))
# plt.barh(features, importances)
# plt.xlabel("Importance")
# plt.title("Feature Importance for Predicting High Insurance Cost")
# plt.gca().invert_yaxis()
# plt.grid(True)
# plt.show()

