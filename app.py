# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 14:32:03 2020

@author: yatharth bansal
"""

from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
import pickle


app=Flask(__name__)


pickle_in=open('Diabetes_Prediction.pkl','rb')
classifier=pickle.load(pickle_in)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=classifier.predict(final_features)
    if prediction==1:
        return render_template('index.html',prediction_text='The person is not diabetic {}'.format(prediction))
    elif prediction==0:
        return render_template('index.html',prediction_text='The person is diabetic {}'.format(prediction))





if __name__=='__main__':
    app.run(debug=True)