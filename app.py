from flask import Flask,render_template,url_for,request
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('RFC.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))

@app.route("/")

def home():
    return render_template('home.html')

@app.route("/predict",methods=["POST","GET"])

def predict():
    data=[float(x) for x in request.form.values()]
    new_Data=scaler.transform(np.array(data).reshape(1,-1))
    final=model.predict(new_Data)[0]
    if final==0:
        return "NO"
    return "yes"






if __name__=="__main__":
    app.run(debug=True)