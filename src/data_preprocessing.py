import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression ,Lasso,ridge_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from src.data_ingestion import ingest_data
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

def get_data()->pd.DataFrame:
    data=ingest_data()
    return data

def drop_columns(data,col_name):
    data.drop(columns=col_name,inplace=True,axis=1)

def encode_data(data)->pd.DataFrame:
    categoried_1=[['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']]
    clarity_mapping = {'I1':0,'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
    color_mapping = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}

    encoder1=OrdinalEncoder(categories=categoried_1)
    encoder2=OrdinalEncoder(categories=clarity_mapping)
    encoder3=OrdinalEncoder(categories=color_mapping)

    data["cut"]=encoder1.fit_transform(data[["cut"]])
    data["clarity"]=data["clarity"].map(clarity_mapping)
    data["color"]=data["color"].map(color_mapping)

def remove_outliers(data,outlier_columns_list):
    for i in outlier_columns_list:
        q3=data[i].quantile(.75)
        q1=data[i].quantile(.25)
        iqr=q3-q1
        lower_bound=q1-1.5*iqr
        upper_bound=q3+1.5*iqr
        data=data[(data[i]>=lower_bound)&(data[i]<=upper_bound)]

def train_test(data)->tuple:
    y=data[["price"]]
    x=data.drop(columns=["price"],axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.4)
    return (x_train,x_test,y_train,y_test)

def scaling(x_train,x_test,y_train,y_test):
    scaler1=MinMaxScaler(feature_range=(0,1))
    x_train_scaled=scaler1.fit_transform(x_train)
    x_test_scaled=scaler1.transform(x_test)
    ##scaler2=MinMaxScaler(feature_range=(0,1))
    ##y_train_scaled=scaler2.fit_transform(y_train)
    ##y_test_scaled=scaler2.transform(y_test)
    
    with open("scaler_x.pkl", "wb") as f:
        pickle.dump(scaler1, f)
    return (x_train_scaled,x_test_scaled,y_train,y_test)

def model_training(a):
    x_train,x_test,y_train,y_test=a[0],a[1],a[2],a[3]
    model=Sequential()
    model.add(Dense(32, activation='relu', input_shape=(9,)))  
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(1,activation='linear'))
    model.compile(optimizer='adam',loss='mse',metrics=['mae'])
    model.fit(x_train,y_train,epochs=12)
    print(r2_score(y_test,model.predict(x_test)))
    print(r2_score(y_train,model.predict(x_train)))

    model.summary()
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)



def clean_data_train_data()->pd.DataFrame:
    data=get_data()
    drop_columns(data,["Unnamed: 0"])
    remove_outliers(data,["carat","depth","table","price","x","y","z"])
    encode_data(data)
    train_test_data=train_test(data)
    data_scaled=scaling(train_test_data[0],train_test_data[1],train_test_data[2],train_test_data[3])
    model_training(data_scaled)
    return data_scaled


clean_data_train_data()

