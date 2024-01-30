import streamlit as st
from utils import *

def main():
    st.title("Weather Predictor")
    ins = WeatherModel()
    prediction = ins.display_next_day()
    st.write("Tomorrow's weather prediction:", prediction)

if __name__ == "__main__":
    main()