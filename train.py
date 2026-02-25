import joblib
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model inside app folder
os.makedirs("app", exist_ok=True)
joblib.dump(model, "app/model.joblib")

print("Model saved successfully!")