import pandas as pd 
from sklearn.linear_model import LinearRegression
import seaborn as sb
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st 
import datetime

st.title('Stock Market prediction')

start_Date = st.date_input('Start Date', value = pd.to_datetime('2016-09-01')) 
end_Date = st.date_input('End Date', value = pd.to_datetime('2022-11-01'))

input = st.text_input('Enter ','GOOGL')
stocks = pdr.DataReader(input, 'yahoo', start_Date, end_Date )


st.subheader('Stocks Data Description')
st.markdown('Information about the historical price of the stock')
st.write(stocks.describe())

st.subheader('Historical Stock Prices')
st.markdown('Stock Prices Changes Over Time')
figure = plt.figure(figsize=(12,6))
plt.plot(stocks['Close'])
plt.xlabel("Date")
plt.ylabel("Closing Price")
st.pyplot(figure)


model = load_model('LSTM_Model.h5')

stocksClose = pd.DataFrame()
stocksClose['Close'] = stocks.Close

#extract the values in the [Close] column 
close_values = stocksClose.values

#80% training size and 20% test size
train_size = math.ceil(len(close_values)* 0.8)

#split the [Close] column to train and test 
train_data = close_values[0:train_size]
test_data = close_values[train_size-60:]

#Normalization using minmaxscaler
scaler = MinMaxScaler()

y_test = test_data[60: ,0]

test_data = scaler.fit_transform(test_data.reshape(-1,1))
x_test = []


#added 60-days from the training size 
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

#Tensorflow accept the data type (Numpy array) 
y_test = np.array(y_test)
x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions using the testing set
stocks_pred = model.predict(x_test)

stocks_pred = scaler.inverse_transform(stocks_pred)

st.subheader('Validation Table')
st.markdown('The close real prices vs the predicted colse prices')
test = stocksClose[train_size:]
validation = pd.DataFrame()
validation['Close'] = pd.DataFrame(test)
validation['Predictions'] = stocks_pred
st.write(validation)


#visulize the reslut for LSTM

st.subheader('Prediction')
st.subheader('Prediction using LSTM')
train = stocksClose[:train_size]
figure2 = plt.figure(figsize=(12,6))
plt.title('Predicted Closing Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')

plt.plot(train)
plt.plot(validation[['Close', 'Predictions']])
plt.legend(['Train', 'Test', 'Pred'], loc='lower right')
st.pyplot(figure2)

#visulize the reslut for Linear Regression' 
st.subheader('Prediction using Linear Regression')
labels = stocks['Close'].copy()
labels = np.array(labels)
labels = labels.reshape(-1,1)
features = stocks['Open'].copy()
features = np.array(features)
features = features.reshape(-1,1)

LinX_train = features[0:train_size]
LinX_test = features[train_size:]
Liny_train = labels[0:train_size]
Liny_test = labels[train_size:]



#Create Linear Regression object
lr_model = LinearRegression()

#Train the model 
lr_model.fit(LinX_train,Liny_train)

#Prediction using our model
pred = lr_model.predict(LinX_test)


figure3 = plt.figure(figsize=(12,6))
Compare = pd.DataFrame({'Orignal values': Liny_test.flatten(), 'Predicted values': pred.flatten()})
plt.title('Predicted Closing Prices')
plt.ylabel('Closing Price')


plt.plot(Compare[['Orignal values', 'Predicted values']])
plt.legend(['Orignal', 'Pred'], loc='lower right')
plt.show()
st.pyplot(figure3)













st.subheader('Next Day Prdection')
st.markdown('The predicted close price for tomorrow is')

days = 1
predict_next_day = []

predict_next_day.append(test_data[len(test_data) + days - 60 : len(test_data)+ days, 0])
    
predict_next_day = np.array(predict_next_day)
predict_next_day = np.reshape(predict_next_day, (predict_next_day.shape[0], predict_next_day.shape[1], 1))

day_pred = model.predict(predict_next_day)
day_pred = scaler.inverse_transform(day_pred)

st.write(day_pred[0][0])
