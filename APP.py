import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

st.set_page_config(
    page_title = "Stock Market Prediction App",
    page_icon = "chart_with_upwards_trend",
    layout="wide"
)

st.title('Stock Market Prediction :chart_with_upwards_trend: :chart_with_downwards_trend:')
user_input = st.text_input("Enter Stock Ticker",'ITC.NS')
itc = yf.Ticker(user_input)
df = (itc.history(period="max"))

#descibing the data
st.subheader('Data from max time period :heavy_dollar_sign: ')
with st.expander("Data Preview: "):
    st.dataframe(df)

fig, fig1 = st.columns(2)
with fig:
    st.subheader("Closing price :vs: Time Graph" )
    fig = plt.figure(figsize = (12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

with fig1:
    st.subheader("Closing Price :vs: 100 Days Moving Avg")
    fig1 = plt.figure(figsize=(12,6))
    plt.plot(df.Close,'b',label = 'Closing Price')
    plt.plot(ma100,'r', label = '100 Days Moving Average')
    plt.legend()
    st.pyplot(fig1)

fig2,fig3 = st.columns(2)

with fig2:
    st.subheader("Closing Price :vs: 200 Days moving Average")
    fig2 = plt.figure(figsize = (12,6))
    plt.plot(df.Close,'r',label = 'Closing Price')
    plt.plot(ma200,'b',label = "200 Days Moving Average")
    plt.legend()
    st.pyplot(fig2)
with fig3:
    st.subheader("Closing Price :vs: 100 Days MA :vs: 200 days MA")
    fig1 = plt.figure(figsize=(12,6))
    plt.plot(df.Close,'b',label = 'Closing Price')
    plt.plot(ma100,'r',label = "100 days moving Avg")
    plt.plot(ma200,'y',label = "200 days moving Avg")
    plt.legend()
    st.pyplot(fig1)



scaler = MinMaxScaler(feature_range=(0,1))

#training
train = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
test = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])
#
#load the model
model = load_model("keras_BE_model_50.h5")


#testing part
past_100_days = train.tail(100)
#final_df = past_100_days.append(test,ignore_index = True)
final_df = pd.concat([past_100_days,train])
ip_data = scaler.fit_transform(final_df)

x_test = []
y_test=[]

for i in range(100, ip_data.shape[0]):
  x_test.append(ip_data[i-100:i])
  y_test.append(ip_data[i, 0])

x_test,y_test = np.array(x_test), np.array(y_test)
y_pred = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

fig4, fig5 = st.columns(2)
with fig4:
    st.subheader(" Original Stock Price :vs: Predicted Stock Price")
    fig4 = plt.figure(figsize=(14,7))
    plt.plot(y_test,'g',label = 'Original Price')
    plt.plot(y_pred,'r',label = 'Predicted Price')
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig4)



df = yf.download(user_input, period="150d")
scaler = MinMaxScaler(feature_range=(0,1))
data = df['Close'].values.reshape(-1, 1)
scaled_data = scaler.fit_transform(data)

# Assuming you have already loaded your pre-trained model
if len(scaled_data) >= 100:
    sequences = []
    for i in range(100, len(scaled_data)):
        sequences.append(scaled_data[i-100:i, 0])
    sequences = np.array(sequences)
    data_min = data.min()
    data_max = data.max()
    sequences = sequences.reshape(sequences.shape[0], sequences.shape[1], 1)

    # Make predictions only if there are enough data points
    y_pred2 = model.predict(sequences)
    y_pred2 = y_pred2.flatten()
    # Reverse scaling to get actual price predictions
    y_pred2 = y_pred2 * (data_max - data_min) + data_min
    
    with fig5:
        st.subheader("LSTM PREDICTED VALUE chart ")
        fig5 = plt.figure(figsize = (14,7))
        plt.plot(y_pred2,'g',label="Predicted Price")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(fig5)

   

    
    y_pred_tomorrow = model.predict(sequences[-1].reshape(1, 100, 1))

    # Reverse scaling to get the actual price prediction
    y_pred_tomorrow = y_pred_tomorrow * (data_max - data_min) + data_min

    # Print or display the predicted price for tomorrow
    st.write("Predicted Stock Closing Price for Tomorrow:", y_pred_tomorrow[0, 0])
    st.snow()

else:
    st.exception("Not enough data points for prediction.")


