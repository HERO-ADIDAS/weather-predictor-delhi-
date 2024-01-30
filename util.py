import time 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge
from torch.utils.data import TensorDataset, DataLoader
import openmeteo_requests
import numpy as np
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import timedelta
import datetime as dt
wmodel=None
now=dt.datetime.now()
date_180_days_ago = now - timedelta(days=155)
str(now.date())

# weather.drop('_id',axis=1,inplace=True)
def nullpecentage(weather):
    null_columns=weather.columns[weather.isnull().any()]
    null_pct=weather[null_columns].isnull().sum()*100/len(weather)
    weather=weather.ffill()
    return null_pct

def convert(weather,target,column):
    weather[target]=weather[column].shift(-1)
    
# def backtest(weather,model,pred,target,start=48,step=12,):
#     all_pred=[] 
#     for i in range(start,weather.shape[0],step):
#         train=weather.iloc[:i,:]
#         test=weather.iloc[i:i+step,:]
#         model.fit(train[pred],train[target])
#         predict=model.predict(test[pred])
#         predict=pd.Series(predict,index=test.index) 
#         combined=pd.concat([predict,test[target]],axis=1)
#         combined.columns=["predict","actual"]
#         combined["diff"]=(combined["predict"]-combined["actual"]).abs()
#         all_pred.append(combined)
#     return pd.concat(all_pred)

def pct_diff(old,new):
    return (new-old)/old
def rolling(weather,horizon,col):
    lablel=f"rolling_{horizon}_{col}"
    weather[lablel]=weather[col].rolling(horizon).mean()
    weather[f"{lablel}_pct"]=pct_diff(weather[lablel],weather[col])
    return weather

def compute_rolling(weather,target,column,pred):
    rolling_horizon=[3,6]
    for horizon in rolling_horizon:
        for col in [column]:
            weather=rolling(weather,horizon,col)
    weather=weather.iloc[6:,:]
    weather.fillna(0,inplace=True)
    pred=weather.columns[~weather.columns.isin([target])]
    return weather
    
def call_pred(weather):
    pred=weather.columns[~weather.columns.isin(["target_temp"])]
    prediction=backtest(weather,wmodel,pred,"target_temp")
    prediction.sort_values("diff",ascending=False)
    
    
class weathermodel:
    def __init__(self):
        
        self.get_weather()
        self.weather = pd.read_csv('data/hourly.csv',index_col="time")
        self.wmodel = Ridge(alpha=0.01)
        self.pred = self.weather.columns[~self.weather.columns.isin(["target_temp"])]
        self.setup()
        self.weather = self.compute_rolling("temperature_2m")
        self.backtest(self.weather, self.wmodel, self.pred, "target_temp")
        
    def backtest(self, weather, model, pred, target, start=48, step=12):
        all_pred=[] 
        for i in range(start,weather.shape[0],step):
            train=weather.iloc[:i,:]
            test=weather.iloc[i:i+step,:]
            model.fit(train[pred],train[target])
            predict=model.predict(test[pred])
            predict=pd.Series(predict,index=test.index) 
            combined=pd.concat([predict,test[target]],axis=1)
            combined.columns=["predict","actual"]
            combined["diff"]=(combined["predict"]-combined["actual"]).abs()
            all_pred.append(combined)
        return pd.concat(all_pred)



    def get_weather(self):
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 28.6358,
            "longitude": 77.2245,
            "hourly": ["temperature_2m", "apparent_temperature", "precipitation_probability", "precipitation", "rain"],
            "timezone": "auto",
            "start_date": str(date_180_days_ago.date()),
            "end_date": str(now.date())
        }
        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        print(f"Coordinates {response.Latitude()}°E {response.Longitude()}°N")
        print(f"Elevation {response.Elevation()} m asl")
        print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
        print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_apparent_temperature = hourly.Variables(1).ValuesAsNumpy()
        hourly_precipitation_probability = hourly.Variables(2).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()
        hourly_rain = hourly.Variables(4).ValuesAsNumpy()

        hourly_data = {"time": pd.date_range(
            start = pd.to_datetime((hourly.Time()), unit = "s"),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}
        hourly_data["temperature_2m"] = hourly_temperature_2m
        hourly_data["apparent_temperature"] = hourly_apparent_temperature
        hourly_data["precipitation_probability"] = hourly_precipitation_probability
        hourly_data["precipitation"] = hourly_precipitation
        hourly_data["rain"] = hourly_rain

        hourly_dataframe = pd.DataFrame(data = hourly_data)
        first_column = hourly_dataframe.columns[0]
        hourly_dataframe.drop(first_column,axis=1)
        hourly_dataframe['time'] = pd.to_datetime(hourly_dataframe['time']) + pd.Timedelta(minutes=30)

        hourly_dataframe.to_csv("data/hourly.csv", index =False)
  
    def setup(self):
        
        convert(self.weather,"target_temp","temperature_2m")
        convert(self.weather,"target_preci","precipitation_probability")
        self.weather = self.weather.ffill()
        nullpecentage(self.weather)
        self.weather.index = pd.to_datetime(self.weather.index)

    def compute_rolling(self, column):
        rolling_horizon=[3,6]
        for horizon in rolling_horizon:
            for col in [column]:
                self.weather=rolling(self.weather,horizon,col)
        self.weather=self.weather.iloc[6:,:]
        self.weather.fillna(0,inplace=True)
        return self.weather

  


    def next_day_prediction(self,weather, model, pred):
        train = weather.iloc[:-1,:]
        test = weather.iloc[-24:,:]
        

        start_time = time.time()
        model.fit(train[pred], train["target_temp"])
        end_time = time.time()
        print(f"Training time: {end_time - start_time} seconds")

        start_time = time.time()
        predict = model.predict(test[pred])
        end_time = time.time()
        print(f"Prediction time: {end_time - start_time} seconds")
        predictions_df = pd.DataFrame({
            'Time Slot': test.index+pd.Timedelta(days=1),
            'Predicted Temp': predict
        })
        
        return predictions_df

    


    def display(self):
        print(self.next_day_prediction(self.weather, self.wmodel, self.pred))

    def display1(self):
        print("the next hour that is ", time.strftime("%H:%M:%S", time.localtime()), "the temperature will be ", self.next_hour_prediction()[0], "°C")








 
