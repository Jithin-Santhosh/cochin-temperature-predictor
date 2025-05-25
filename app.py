import streamlit as st
import pandas as pd
import datetime
import joblib


model=joblib.load('temp_model.pkl')

st.title("🌤️TEMPERATURE PREDICTOR     🌤️")
st.write("Select a date to predict the temperature for :")
selected_date=st.date_input("Pick a date",datetime.date.today())

if selected_date:
    day=selected_date.day
    month=selected_date.month
    year=selected_date.year

    input_features=pd.DataFrame([[day, month, year]], columns=['Day', 'Month', 'Year'])
    prediction = model.predict(input_features)[0]

    st.subheader(f"📅 Predicted Max Temperature for {selected_date}:")
    st.success(f"{prediction:.2f} °C")
