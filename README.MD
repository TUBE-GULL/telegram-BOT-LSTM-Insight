<h1 align="center">telegram-BOT-LSTM-Insight</h1>

<h2 align="center">Used Libraries</h2>
<div align="center">
 <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="60" height="60"/>

<a href="https://numpy.org/" target="_blank" rel="noreferrer"><img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original-wordmark.svg" title="Numpy" alt="Numpy" width="60" height="60"/> </a>

<a href="https://www.tensorflow.org/" target="_blank" rel="noreferrer"><img src="https://github.com/devicons/devicon/blob/master/icons/tensorflow/tensorflow-original.svg" title="tensorflow" alt="tensorflow" width="60" hight="60"> </a>

<a href="https://keras.io/" target="_blank" rel="noreferrer"><img src="https://github.com/devicons/devicon/blob/master/icons/keras/keras-original.svg" title="keras" alt="keras" width="60" hight="60"> </a>

<a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"><img src="https://github.com/devicons/devicon/blob/master/icons/pandas/pandas-original.svg" title="Pandas" alt="Pandas" width="60" height="60"/> </a>

 <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"><img src="https://github.com/devicons/devicon/blob/master/icons/scikitlearn/scikitlearn-original.svg" title="sklearn" alt="sklearn" width="60" height="60"/></a>

 <a href="https://matplotlib.org/" target="_blank" rel="noreferrer"><img src="https://github.com/devicons/devicon/blob/master/icons/matplotlib/matplotlib-original.svg" title="plotly" alt="pltly" width="60" height="60"/></a>

 <a href="https://plotly.com/" target="_blank" rel="noreferrer"><img src="https://github.com/devicons/devicon/blob/master/icons/plotly/plotly-original.svg" title="plotly" alt="pltly" width="60" height="60"/></a>

<a href="https://docs.aiogram.dev/en/latest/index.html" target="aiogram"><img src="https://docs.aiogram.dev/en/latest/_static/logo.png" alt="aiogram" width="60" height="60"/></a>

</div>


## Description

This project is a Telegram bot that uses an LSTM (Long Short-Term Memory) model to forecast prices for the next 7 days. The bot allows users to get price predictions for stocks, indices, cryptocurrencies, and other financial assets.

## Features

- **Price Forecasting**: Utilizes an LSTM model to predict prices 7 days into the future.
- **Telegram Bot**: An interactive bot in Telegram that lets users request forecasts for various assets.
- **Support for Various Assets**: Forecasting for stocks, indices, cryptocurrencies, and other types of financial instruments.


<h2 align="center">Instructions</h2>

```bash
1 /start - start bot 

2 /forecast "your choice" - specify the asset after the command. For example: /forecast TSLA"

3 /info - list of what can be requested

4 /help - set commands to the bot
```

## Installation dependencies

````bash
# install package
$ pip install aiogram
$ pip install numpy
$ pip install pandas
$ pip install pandas-ta
$ pip install yfinance
$ pip install scikit-learn
$ pip install tensorflow

````
## Running the app

```bash
# start in linux and mac
$ python3 main.py

# start in windows 
$ python main.py
```

##P.S

<P>In my opinion, the project serves as a solid foundation for analyzing stock and other markets. Existing trading strategies can be adapted and applied to this model, allowing for more comprehensive and accurate predictions. Since the model has not yet been thoroughly developed due to time constraints, there is potential for further improvement and optimization in the future.</P>