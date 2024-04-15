import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)
model = pickle.load(open("DT.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]

    prediction = model.predict(features) # Assuming binary classification

    if prediction == 0:
        prediction_text = "You are at low risk for heart disease."  # More informative
    else:
        prediction_text = "You are at higher risk for heart disease. Please consult a doctor for further evaluation."  # Emphasize seeking medical advice

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)