# models 
from add_indicator_df import add_indicator_df
from add_target_df import add_target_df
from preparing_data_for_lstm import preparing_data_for_lstm
from typing import Tuple, Optional
from copy import deepcopy 
import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt


import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def training_model(df, features, target, SEQ_LENGTH, name_save_model=None,
                   lstm_units1=150, lstm_units2=100, dense_units=50, 
                   dropout_rate=0.2, epochs=50, batch_size=32, grade=False,
                   rsi: bool = True,
                   ema_20: bool = True,
                   ema_100: bool = True,
                   ema_150: bool = True,
                   wma_20: bool = True,
                   wma_100: bool = True,
                   wma_150: bool = True,
                   BB: bool = True):
    """
    Функция для обучения модели LSTM на основе предоставленного DataFrame с финансовыми данными.
    
    Параметры:
    - df (pd.DataFrame): Исходный DataFrame с финансовыми данными.
    - features (list): Список имен признаков для использования в модели.
    - target (str): Имя целевой переменной.
    - SEQ_LENGTH (int): Длина временного окна для LSTM.
    - lstm_units1 (int): Количество единиц в первом слое LSTM.
    - lstm_units2 (int): Количество единиц во втором слое LSTM.
    - dense_units (int): Количество единиц в слое Dense.
    - dropout_rate (float): Процент Dropout.
    - epochs (int): Количество эпох для обучения.
    - batch_size (int): Размер мини-батча.
    - grade (bool): Флаг для оценки модели после обучения.
    - rsi (bool): Использовать индикатор RSI.
    - ema_20 (bool): Использовать 20-дневную EMA.
    - ema_100 (bool): Использовать 100-дневную EMA.
    - ema_150 (bool): Использовать 150-дневную EMA.
    - wma_20 (bool): Использовать 20-дневную WMA.
    - wma_100 (bool): Использовать 100-дневную WMA.
    - wma_150 (bool): Использовать 150-дневную WMA.
    - BB (bool): Использовать полосы Боллинджера.

    Возвращаемые значения:
    - history (History): Объект History, содержащий информацию о процессе обучения.
    - X_train (np.array): Обучающая выборка.
    - X_test (np.array): Тестовая выборка.
    - y_train (np.array): Целевая переменная для обучения.
    - y_test (np.array): Целевая переменная для тестирования.
    """
    
    # Подготовка данных
    data = add_indicator_df(add_target_df(df), RSI=rsi, EMA_20=ema_20, EMA_100=ema_100,
                            EMA_150=ema_150, WMA_20=wma_20, WMA_100=wma_100,
                            WMA_150=wma_150, Bollinger_Bands=BB)
    
    # Подготовка данных для LSTM
    X_train, X_test, y_train, y_test = preparing_data_for_lstm(data, features, target, SEQ_LENGTH)

    # Создание модели LSTM
    model = Sequential([
        LSTM(lstm_units1, activation='tanh', return_sequences=True, input_shape=(SEQ_LENGTH, len(features))),
        Dropout(dropout_rate),
        LSTM(lstm_units2, activation='tanh'),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Определение EarlyStopping и ReduceLROnPlateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

    # Обучение модели
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, lr_schedule])
    
    def evaluate_model():
        # Оценка модели
        loss = model.evaluate(X_test, y_test)
        print(f'Test loss: {loss}')

        # Предсказания
        predictions = model.predict(X_test)

        # Визуализация результатов    
        plt.figure(figsize=(14, 7))
        plt.plot(data.index[SEQ_LENGTH + len(X_train):], y_test, label='Actual')
        plt.plot(data.index[SEQ_LENGTH + len(X_train):], predictions, label='Predicted')
        plt.legend()
        plt.show()
        
    # Оценка модели при необходимости
    if grade:
        evaluate_model()
    
    # Сохранение модели
    if name_save_model is None:
        model.save('model1.keras')
    else:    
        model.save(f'{name_save_model}.keras')
        
    return history, X_train, X_test, y_train, y_test


# Определите размер временного окна (например, 10 дней)
SEQ_LENGTH = 10

# Выберите признаки и цель
features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'RSI', 'EMA_20', 'EMA_100', 'EMA_150', 'WMA_20', 'WMA_100', 'WMA_150']
target = 'TargetNextClose'

# Акции:
# AAPL — Apple Inc.
# MSFT — Microsoft Corporation
# GOOGL — Alphabet Inc. (Class A)
# AMZN — Amazon.com Inc.
# TSLA — Tesla Inc.
# META — Meta Platforms Inc. (ранее Facebook)
# NFLX — Netflix Inc.
# NKE — Nike Inc.
# BABA — Alibaba Group Holding Ltd.
# NVDA — NVIDIA Corporation
# Индексы:
# ^GSPC — S&P 500
# ^DJI — Dow Jones Industrial Average
# ^IXIC — NASDAQ Composite
# ^RUT — Russell 2000
# ^FTSE — FTSE 100 (Лондонский фондовый индекс)
# ^DAX — DAX (Немецкий фондовый индекс)
# Криптовалюты:
# BTC-USD — Bitcoin (BTC) к USD
# ETH-USD — Ethereum (ETH) к USD
# ADA-USD — Cardano (ADA) к USD
# SOL-USD — Solana (SOL) к USD
# DOGE-USD — Dogecoin (DOGE) к USD
# XRP-USD — Ripple (XRP) к USD
# Товары:
# GC=F — Gold Futures
# CL=F — Crude Oil Futures
# SI=F — Silver Futures
# Облигации и другие активы:
# ^IRX — 3-Month Treasury Bill
# ^TNX — 10-Year Treasury Note Yield

#crypto
data_btc = yf.download(tickers='BTC-USD', start='1999-01-01', end='2024-06-10') # BTC-USD — Bitcoin (BTC) к USD
# data_eth = yf.download(tickers='ETH-USD', start='1999-01-01', end='2024-06-10') # ETH-USD — Ethereum (ETH) к USD
#index
data_spx = yf.download(tickers='^GSPC', start='1999-01-01', end='2024-06-10') # BTC-USD — Bitcoin (BTC) к USD
# data_nasdaq = yf.download(tickers='^IXIC', start='1999-01-01', end='2024-06-10') # ETH-USD — Ethereum (ETH) к USD
#shares
data_aapl = yf.download(tickers='AAPL', start='1999-01-01', end='2024-06-10') # AAPL — Apple Inc.
# data_tsla = yf.download(tickers='TSLA', start='1999-01-01', end='2024-06-10') # TSLA — Tesla Inc.


history_btc, X_train, X_test, y_train, y_test  = training_model(data_btc, features, target, SEQ_LENGTH, epochs=100, name_save_model='model_crypto',grade=True)
history_shares, X_train, X_test, y_train, y_test  = training_model(data_aapl, features, target, SEQ_LENGTH, epochs=100, name_save_model='model_shares',grade=True)
history_index, X_train, X_test, y_train, y_test  = training_model(data_spx, features, target, SEQ_LENGTH, epochs=100, name_save_model='model_index',grade=True)


# class LSTMModel(): 
#     def __init__(self, features, target, seq_length, lstm_units1=150, lstm_units2=100, dense_units=50, 
#                  dropout_rate=0.2, epochs=50, batch_size=32):
#         self.features = features
#         self.target = target
#         self.seq_length = seq_length
#         self.lstm_units1 = lstm_units1
#         self.lstm_units2 = lstm_units2
#         self.dense_units = dense_units
#         self.dropout_rate = dropout_rate
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.model = None
#         self.history = None
