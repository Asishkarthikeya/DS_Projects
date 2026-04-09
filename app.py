from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import joblib
import pickle
import sklearn


app = Flask(__name__)


@app.route('/')
def hello_world():
	return render_template('index.html')

@app.route('/about')
def about():
     return render_template('about.html')

@app.route('/team')
def team():
     return render_template('team.html')

@app.route('/covid')
def covid():
    return render_template('covid.html')


@app.route('/breastcancer')
def breast_cancer():
    return render_template('breast.html')


@app.route('/braintumor')
def brain_tumor():
    return render_template('braintumor.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/alzheimer')
def alzheimer():
    return render_template('alzheimer.html')


@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')


@app.route('/heart')
def heartdisease():
    return render_template('heart.html')

@app.route('/resulth',methods=['POST'])
def resulth():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        nmv = float(request.form['nmv'])
        tcp = float(request.form['tcp'])
        eia = float(request.form['eia'])
        thal = float(request.form['thal'])
        op = float(request.form['op'])
        mhra = float(request.form['mhra'])
        age = float(request.form['age'])
        print(np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
    
    with open('models/heart.pkl', 'rb') as file:
        model = pickle.load(file)
    pred = model.predict(np.array([age, tcp, mhra, eia, op, nmv, thal]).reshape(1, -1))
    return render_template('resulth.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

@app.route('/resultd', methods=['POST'])
def resultd():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bloodpressure = float(request.form['bloodpressure'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetespedigree = float(request.form['diabetespedigree'])
        age = float(request.form['age'])
        skinthickness = float(request.form['skin'])
        
        with open('models/diabetes.pkl', 'rb') as file:
            diabetes_model = pickle.load(file)
        pred = diabetes_model.predict(np.array([pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age]).reshape(1, -1))
        return render_template('resultd.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)
        

@app.route('/resultbc', methods=['POST'])
def resultbc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = float(request.form['age'])
        cpm = float(request.form['concave_points_mean'])
        am = float(request.form['area_mean'])
        rm = float(request.form['radius_mean'])
        pm = float(request.form['perimeter_mean'])
        cm = float(request.form['concavity_mean'])
        
        with open('models/breast.pkl', 'rb') as file:
            model = pickle.load(file)
        pred = model.predict(np.array([cpm, am, rm, pm, cm]).reshape(1, -1))
        return render_template('resultbc.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

if __name__ == '__main__':
	app.run()

