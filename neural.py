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


class Neural:

    def __init__(self, choices):
        global choice
        #reception des variables
        choice = choices
        crypto_currency = choices[0]
        against_currency = choices[1]
        futur_day = int(choices[2])
        mod = choices[3]

        #choix de variables internes

        start = dt.datetime(2020, 1, 1)
        end = dt.datetime.now()
        prediction_days = 60
        today = end.strftime("%B %d, %Y")
        folder = os.getcwd()
        epoch = 60

        #telechargement et affichage des donnes a partir de yahoo finance

        data = yf.download(f'{crypto_currency}-{against_currency}', start=start, end=end)

        print(data.head())

        #creation du scaler pour preparer les donnes

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        #print(scaled_data)

        #preparation des donnes d'entrainement

        x_train, y_train = [], []

        #ce block d'instruction n'est lancer que si on choisi de cree un model

        if choices[3] in ["True", True]:
            for x in range(prediction_days, len(scaled_data) - futur_day):
                x_train.append(scaled_data[x - prediction_days:x, 0])
                y_train.append(scaled_data[x + futur_day, 0])

            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            #creation du reseau de neurones

            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))

            model.compile(loss='mean_squared_error', optimizer='adam')

            #affecter les donnes d'entrainement au model

            model.fit(x_train, y_train, epochs=epoch, batch_size=32)

            #sauvegarde du model

            model.save(f"models/{crypto_currency}-{against_currency} in {futur_day} days.model")


        #chargement du model au cas ou on a choisi de ne pas cree un nouveau

        model = keras.models.load_model(f"models/{crypto_currency}-{against_currency} in {futur_day} days.model")

        #preparation des dates de donnees de test

        test_start = dt.datetime(2022, 1, 1)
        test_end = dt.datetime.now()

        # preparation des donnees de test

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


        #utiliser le model cree pour faire des predictions

        prediction_prices = model.predict(x_test)
        prediction_prices = scaler.inverse_transform(prediction_prices)

        #creation des graphiques

        plt.clf()
        plt.plot(actual_prices, color='black', label='actual prices')
        plt.plot(prediction_prices, color='green', label='Predicted prices')
        plt.title(f'{crypto_currency} in {against_currency} prediction in {futur_day} days')
        plt.xlabel('Time')
        plt.ylabel('Price')

        #sauveharde des graphiques

        plt.savefig(f'imgs/{today}-{crypto_currency} {futur_day} days.png', dpi=1200)
        # plt.show()

        #preparation de la prediction futur

        real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

        #retransformer les donnes modifier par le scaler

        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)



        yesterday = dt.datetime.now() - dt.timedelta(days=1)


        #preparation de l'affichage finale

        crypto_price_now = yf.download(f'{crypto_currency}-{against_currency}', start=yesterday, end=end)
        crypto_price_now = crypto_price_now['Open'].values
        # print(type(crypto_price_now))
        # print(type(prediction))
        print(f'today {crypto_currency}\'s price is {crypto_price_now}')
        print(f'in {futur_day} days i predict {prediction}')

        day = futur_day
        resultat = ' '.join(map(str, crypto_price_now))
        resultat1 = str(prediction).replace('[', '').replace('[', '').replace(']', '')

        #lancement de la boite de dialogue contenant les resultats finaux

        window = tk.Tk()
        window.geometry("350x200")
        window.title("Result interface")
        window.iconbitmap("icon.ico")
        frame = tk.Frame(window, border=1)
        
        frame.pack(pady=20, padx=20, fill="both", expand=True)

        label1 = tk.Label(master=frame)
        label1.place(x=40, y=30)
        label1.config(text="Today price : "+resultat+" "+choices[1])
        label2 = tk.Label(master=frame)
        label2.place(x=40, y=80)
        label2.config(text="Price after "+str(day)+" days : "+resultat1+" "+choices[1])
        but = tk.Button(master=frame, width=20, text="Show Graph", command=self.graph)
        but.place(x=55, y=120)

    def graph(self):
        global choice
        today = (dt.datetime.now()).strftime("%B %d, %Y")
        crypto_currency = choice[0]
        futur_day = int(choice[2])
        img = Image.open(f'imgs/{today}-{crypto_currency} {futur_day} days.png')
        img.show()




