import pandas as pd

full_data = pd.read_csv(r"C:\Users\Sam\Documents\ML\sp500_repo\SP500_Predictor\full_data.csv")
full_data_ml = pd.DataFrame(full_data)

X_lagged = full_data_ml[['CPI', 'Unemployment_Rate', 'G10Yield', 'G5Yield', 'G3Yield', 'EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY']]
y_lagged = full_data_ml['SP500_Close']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, max_error

X_lagged = full_data_ml[['CPI', 'Unemployment_Rate', 'G10Yield', 'G5Yield', 'G3Yield', 'EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY']]
y_lagged = full_data_ml['SP500_Close']

X_train, X_test, y_train, y_test = train_test_split(X_lagged, y_lagged, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

import streamlit as st

# Streamlit UI
st.title("S&P 500 Price Predictor")

st.write("""
This app predicts the **S&P 500** closing price based on various economic indicators.
""")

# User input fields
CPI = st.number_input('CPI (Consumer Price Index)', value=0.0)
Unemployment_Rate = st.number_input('Unemployment Rate (%)', value=0.0)
G10Yield = st.number_input('G10 Yield (%)', value=0.0)
G5Yield = st.number_input('G5 Yield (%)', value=0.0)
G3Yield = st.number_input('G3 Yield (%)', value=0.0)
EURUSD = st.number_input('EUR/USD Exchange Rate', value=0.0)
GBPUSD = st.number_input('GBP/USD Exchange Rate', value=0.0)
USDCHF = st.number_input('USD/CHF Exchange Rate', value=0.0)
USDJPY = st.number_input('USD/JPY Exchange Rate', value=0.0)

if st.button('Predict'):
    input_data = pd.DataFrame([[CPI, Unemployment_Rate, G10Yield, G5Yield, G3Yield, EURUSD, GBPUSD, USDCHF, USDJPY]], 
                              columns=['CPI', 'Unemployment_Rate', 'G10Yield', 'G5Yield', 'G3Yield', 'EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY'])
    prediction = model.predict(input_data)
    st.write(f"Predicted S&P 500 Closing Price Jan 1st Next Month: ${prediction[0]:,.2f}")