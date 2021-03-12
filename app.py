import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
import gunicorn
import pickle

app = Flask(__name__)

model = pickle.load(open("logreg_model.pkl", "rb"))

@app.route('/')
def route():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == "POST":
        child_disease = request.form['child_disease']
        trauma = request.form['trauma']
        surgical = request.form['surgical']
        alcohol = request.form['alcohol']
        smoking = request.form['smoking']
        sitting_hours = request.form['sitting_hours']
        
        peluang_normal = round(model.predict_proba([[child_disease, trauma, surgical,
                         alcohol, smoking, sitting_hours]])[0,0] * 100, 2)
        results = 'You Have {}% Probabilities of Having an Altered Fertility'.format(peluang_normal)

        # result = model.predict([[child_disease, trauma, surgical,
        #                 alcohol, smoking, sitting_hours]])[0]
        # results = 'You are going to have {} fertility'.format(result)
        return render_template('result.html', results=results)
    return render_template('predict.html')
 
if __name__ == '__main__':
    app.run(debug=True)