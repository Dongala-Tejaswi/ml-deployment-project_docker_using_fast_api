from fastapi import FastAPI
import joblib
import numpy as np
import os

# Load model safely
model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
model = joblib.load(model_path)

# Class names
class_names = np.array(['setosa', 'versicolor', 'virginica'])

# Create FastAPI app
app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Iris model API'}

@app.post('/predict')
def predict(data: dict):
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    class_name = class_names[prediction][0]
    return {'predicted_class': class_name}