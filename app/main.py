# app/main.py
 
from flask import Flask, request, jsonify

import joblib

import pandas as pd

import os
 
app = Flask(__name__)
 
# Load the model from the correct path

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

model = joblib.load(model_path)
 
@app.route("/predict", methods=["POST"])

def predict():

    data = request.get_json(force=True)

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return jsonify({"prediction": int(prediction[0])})
 
@app.route("/health", methods=["GET"])

def health():

    return "✅ API is live and working!"
 
if __name__ == "__main__":

    app.run(debug=True, host="0.0.0.0", port=5000)

 