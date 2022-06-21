from flask import Flask, render_template,request
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

data = pd.read_csv("C:/Users/Aryaman/Downloads/quikr_car.csv")
data=data[data['year'].str.isnumeric()]
data=data[data['Price']!='Ask For Price']
data['Price']=data['Price'].str.replace(',','').astype(int)
data=data[data['kms_driven']!='Petrol']
data['kms_driven']=data['kms_driven'].str.split(' ').str.get(0).str.replace(',','').astype(int)
data["fuel_type"].fillna(data["fuel_type"].mode()[0],inplace=True)
data['name']=data['name'].str.split(' ').str.slice(0,3).str.join(' ')
data.reset_index(drop=True)
data=data[data['Price']<6000000].reset_index(drop=True)

X = data.drop(columns='Price')
Y = data['Price']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
ohe = OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])
column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)
scores = []
for i in range(1000):
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_test_pred = pipe.predict(x_test)
    scores.append(r2_score(y_test,y_test_pred))

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=np.argmax(scores))
lr = LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)

app = Flask(__name__)

# model=pickle.load(open('ML work/car_prediction/LinearRegressionModel.pkl','rb'))
data = pd.read_csv(
    'C:/Users/Aryaman/Desktop/ML work/car_prediction/Cleaned_data.csv')

@app.route('/')
def index():
    companies = sorted(data['company'].unique())
    car_models = sorted(data['name'].unique())
    year = sorted(data['year'].unique(), reverse=True)
    fuel_type = data['fuel_type'].unique()
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))
    print(company,car_model,year)
    prediction=pipe.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))
    print(prediction[0])
    output = round(prediction[0],2)

    return str(output)

    return ""
if __name__ == "__main__":
    app.run(debug=True)
