from flask import Flask,redirect,url_for,render_template,request
import pickle as pkl
import pandas as pd
app=Flask(__name__)
@app.route('/')
def welcome():
    return render_template('index.html',output='No prediction made yet')
@app.route('/predict',methods=['GET','POST'])
def predict():
    output="no result"
    if request.method=='POST':
        
        list_data=[]
        for x,y in request.form.items():
            list_data.append(float(y))
        print("these are all keys",request.form)
        data=pd.DataFrame([list_data],columns=request.form.keys())
        print(data)
        model=pkl.load(open('log_regression_model_new.pkl','rb'))
        output_value=model.predict(data)[0]
        if output_value==0:
            output="The breast cancer is benevolent"
        if output_value==1:
            output="The breast cancer is malignant"
    return render_template('index.html',output=output)


if __name__=='__main__':
    app.run(debug=True) 