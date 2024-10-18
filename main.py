import sys
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
from neural import Neural

def get_intel(intel):
    network = Neural(intel)
    return intel


def exit():

    return 0



root = tk.Tk()
app = Interface(root,get_intel,exit)

choices = app.intel

root.mainloop()