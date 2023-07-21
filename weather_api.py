import requests
from dotenv import load_dotenv
import os
from dataclasses import dataclass

load_dotenv()

api_key = os.getenv('API_KEY')

@dataclass
class WeatherData():

    Temperature : float


def get_lat_lon(city_name,API_key):

    response = requests.get(f'http://api.openweathermap.org/geo/1.0/direct?q={city_name}&appid={API_key}').json()
    
    data = response[0]
    lat,lon = data.get('lat'),data.get('lon')
    return lat,lon


def current_weather(lat,lon,API_key):
    response = requests.get(f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_key}&units=metric').json()

    data = WeatherData(
        Temperature= response.get('main').get('temp')
    )
    return data

def main(city_name):
    try:

        lat,lon = get_lat_lon(city_name,api_key)
        weatherdata = str(current_weather(lat,lon,api_key))
        weatherdata = weatherdata[12:28]
        degree_sign = u"\N{DEGREE SIGN}"
   
        return f'{weatherdata}{degree_sign}C'
    except:
        return "Please check the location and try again"
