from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('Descision_tree.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        dept=int(request.form['department'])
        indoor=int(request.form['indoor'])
        outdoor=int(request.form['outdoor'])
        wake_up=int(request.form['wake_up'])
        sleep_time=int(request.form['sleep_time'])
        introvert=int(request.form['introvert'])
        extrovert=int(request.form['extrovert'])
        conservative=int(request.form['conservative'])
        neat=int(request.form['neat'])
        Food=int(request.form['Food'])
        language=int(request.form['language'])
        prediction=model.predict([[dept,indoor,outdoor,wake_up,sleep_time,introvert,extrovert,conservative,neat,Food,language]])
        output=prediction
        return render_template('index.html',prediction_text="Your compatible {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
