from flask import Flask, request, render_template,flash,redirect,session,abort,jsonify
import os
import pickle
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy
#model = joblib.load("modelsvm.pkl")
app = Flask(__name__)

model = joblib.load("modelpred.pkl")
#pipe = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', model)])
@app.route('/')
def root():
    return render_template('index.html')
@app.route('/welcome')
def welcome():
    return render_template('index1.html')
@app.route('/predict', methods=["POST"])
def predict():
    input_features = [i for i in request.form.values()]
    input_numpy = [numpy.array(input_features)]
    prediction = model.predict(input_numpy)
    if prediction[0] >= 0 and prediction[0] <= 50:
        value = prediction[0]
        result = 'The estimated AQI is Good!'
    if prediction[0] >= 51 and prediction[0] <= 100:
        value = prediction[0]
        result = 'The estimated AQI is Satisfactory!'
    if prediction[0] >= 101 and prediction[0] <= 200:
        value = prediction[0]
        result = 'The estimated AQI is Moderate!'
    if prediction[0] >= 201 and prediction[0] <= 300:
        value = prediction[0]
        result = 'The estimated AQI is Poor!'
    if prediction[0] >= 301 and prediction[0] <= 400:
        value = prediction[0]
        result = 'The estimated AQI is Very Poor!'
    if prediction[0] >= 401 and prediction[0] <= 500:
        value = prediction[0]
        result = 'The estimated AQI is Severe!'

    return render_template("result.html", result=result,value=value)
app.secret_key = os.urandom(12)
app.run(port=5500, host='0.0.0.0', debug=True)