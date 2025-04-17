import nasdaqdatalink as ndl
import numpy as np 
import pandas as pd 
import yfinance as yf
from pandas_datareader import data as web
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
import streamlit as st





start = '2015-01-01'
end = '2025-04-01'
st.title("Flick Finance")
user_input = st.text_input('Enter Stock Ticker')

df = yf.download(user_input, start=start, end=end)
df = df.reset_index()
# print(df.drop(columns=['Date']))

st.subheader('Data from 2015 - 2025')
st.write(df.describe())

st.subheader('Closing Price vs Time')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time with 100 Days Moving Average')
fig2 = plt.figure(figsize=(12,6))
plt.plot(df.Close.rolling(window=100).mean())
plt.plot(df.Close, 'g')
st.pyplot(fig2)


st.subheader('Closing Price vs Time with 100 & 200 Days Moving Average')
fig3 = plt.figure(figsize=(12,6))
plt.plot(df.Close.rolling(window=100).mean(), 'r')
plt.plot(df.Close.rolling(window=200).mean(), 'b')
plt.plot(df.Close, 'g')
st.pyplot(fig3)

# # print(plt.plot(df.Close))
# avg100 = df.Close.rolling(window=100).mean()
# # print(avg100)
# # plt.figure(fig=(12,6))
# # plt.plot(df.Close)
# plt.plot(avg100, 'r')
# avg200 = df.Close.rolling(window=200).mean()
# # print(avg200)
# plt.plot(avg200, 'g')
# # print(df.shape)


train = pd.DataFrame(df['Close'][:int(len(df)*0.7)])
test = pd.DataFrame(df['Close'][int(len(df)*0.7):])
# # print(train.shape)
# # print(test.shape)
scaler = MinMaxScaler(feature_range=(0,1))
data_train = scaler.fit_transform(train)

# #Splitting Data into X-TRAIN AND Y-TRAIN
# x_train = []
# y_train = []
# for i in range(100, data_train.shape[0]):
#     x_train.append(data_train[i-100:i])
#     y_train.append(data_train[i,0])

# numpy_x_train = np.array(x_train)
# numpy_y_train = np.array(y_train)

#Load Model
model = load_model('./stocks_model.keras')

# print(numpy_x_train.shape)

# model = Sequential()
# model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(numpy_x_train.shape[1],1)))
# model.add(Dropout(0.2))

# model.add(LSTM(units=60, activation='relu', return_sequences=True))
# model.add(Dropout(0.3))

# model.add(LSTM(units=80, activation='relu', return_sequences=True))
# model.add(Dropout(0.4))

# model.add(LSTM(units=120, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(units=1))

# print(model.summary())

# # model.compile(optimizer='adam', loss='mean_squared_error')
# # model.fit(numpy_x_train, numpy_y_train, epochs=50)
# # model.save('stocks_model.keras')


#Testing Here
past100 = train.tail(100).copy()
# print(past100)
finalDf = pd.concat([past100,test], ignore_index=True)
# print(finalDf.head())
input_data = scaler.fit_transform(finalDf)
# print(input_data.shape)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
print(x_test.shape)
print(y_test.shape)

y_predicted = model.predict(x_test)
print(y_predicted.shape)
after = scaler.scale_
# print(after)
scale_factor = 1/after[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Predictions vs Actual Price Graph')
fig4 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

# Get RMSE

rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
st.write("Root Mean Squared Error (RMSE):", rmse)

# Get R² Score
r2 = r2_score(y_test, y_predicted)
st.write("R² Score:", r2)

