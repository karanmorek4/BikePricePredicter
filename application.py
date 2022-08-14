from flask import Flask, render_template,request
import pandas as pd
import numpy as np
import pickle


app=Flask(__name__)
car=pd.read_csv("Used_Bikes.csv")

no_city=list(car.city.unique())
no_brand=list(car.brand.unique())

model = pickle.load(open("RandomForestRegressor.pkl" , 'rb'))


@app.route('/')
def index():
    brand=sorted(car.brand.unique())
    power=sorted(car.power.unique())
    city=sorted(car.city.unique())
    age_of_bike=sorted(car.age.unique())
    owner=sorted(car.owner.unique())
    model=sorted(car.bike_name.unique())
    return render_template('index.html',brands=brand,citys=city,age_of_bikes=age_of_bike,owners=owner,powers=power, models=model)

@app.route('/predict' , methods=['POST'])
def predict():
    global no_city,no_brand

    km=float(request.form.get('kilo_driven'))
    own=request.form.get('Owner')
    age=int(request.form.get('year'))
    pow=int(request.form.get('power'))
    city=request.form.get('city')
    brand=request.form.get('brand')
    if own == 'First Owner' :
        own=int(0)
    elif own == 'Second Owner' :
        own=int(1)
    elif own == 'Third Owner' :
        own=int(2)
    elif own == 'Fourth Owner Or More' :
        own=int(3)

    city=no_city.index(city)
    brand=no_brand.index(brand)

    prediction  = model.predict(pd.DataFrame([[km,own,age,pow,city,brand]] , columns=['kms_driven','owner','age','power','city_code','brand_code']))
    prediction = prediction.tolist()
    return str(np.round(prediction[0],2))

if __name__ == '__main__':
    app.run(debug=True)