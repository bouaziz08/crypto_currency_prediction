import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import tensorflow
from tensorflow import keras
from keras.layers import Dense, Dropout,LSTM
from keras.models import Sequential
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
from GUI import Interface
import tkinter as tk
from tkinter import messagebox
import glob
from PIL import Image

crypto_money = ['ETH', 'DOGE']
currency = ['USD', 'EUR']
days = [1, 2, 3, 4, 5]


def make(choices):
    crypto_currency = choices[0]
    against_currency = choices[1]
    futur_day = int(choices[2])


    start = dt.datetime(2020, 1, 1)
    end = dt.datetime.now()
    prediction_days = 60
    today = end.strftime("%B %d, %Y")
    folder = os.getcwd()
    epoch = 25


    data = yf.download(f'{crypto_currency}-{against_currency}', start=start, end=end)

    #print(data.head())

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    x_train, y_train = [], []
    # if choices[3] in ["True", True]:
    for x in range(prediction_days, len(scaled_data) - futur_day):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x + futur_day, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(x_train, y_train, epochs=epoch, batch_size=32)

    model.save(f"models/{crypto_currency}-{against_currency} in {futur_day} days.model")

    model = keras.models.load_model(f"models/{crypto_currency}-{against_currency} in {futur_day} days.model")

    test_start = dt.datetime(2022, 1, 1)
    test_end = dt.datetime.now()

    test_data = yf.download(f'{crypto_currency}-{against_currency}', test_start, test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)

    plt.plot(actual_prices, color='black', label='actual prices')
    plt.plot(prediction_prices, color='green', label='Predicted prices')
    plt.title(f'{crypto_currency} in {against_currency} prediction in {futur_day} days')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.savefig(f'imgs/{today}-{crypto_currency}-{against_currency} {futur_day} days.png', dpi=1200)
    # plt.show()

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    yesterday = dt.datetime.now() - dt.timedelta(days=1)

    crypto_price_now = yf.download(f'{crypto_currency}-{against_currency}', start=yesterday, end=yesterday)
    crypto_price_now = crypto_price_now['Close'].values
    # print(type(crypto_price_now))
    # print(type(prediction))
    print(f'today {crypto_currency}\'s price is {crypto_price_now}')
    print(f'in {futur_day} days i predict {prediction}')

for crypto in crypto_money:
    for cur in currency:
        for day in days:
            choices = [crypto, cur, day]
            print(choices)
            make(choices)