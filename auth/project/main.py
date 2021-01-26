from flask import Blueprint, render_template 
from flask_login import login_required, current_user
from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

main = Blueprint('main', __name__)

app = Flask(__name__)

model = load_model('deployment_110620201')
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@main.route('/')
@login_required
def home():
    return render_template("home.html")

@main.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Label[0])
    return render_template('home.html',pred='Expected Cost will be Ksh {}'.format(prediction))

@main.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

@main.route('/index')
def index():
    return render_template('index.html')

@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', name=current_user.name)
