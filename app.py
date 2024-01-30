import streamlit as st
from utils import *

def main():
    st.title("Weather Predictor")
    ins = WeatherModel()
    predictio = ins.display_next_day()
    st.write("Tomorrow's weather prediction:")
    st.write(predictio)
if __name__ == "__main__":
    main()