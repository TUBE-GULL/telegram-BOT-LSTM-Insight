from copy import deepcopy
import pandas as pd
import pandas_ta as ta
from typing import Tuple


def add_indicator_df(
    df: pd.DataFrame,
    RSI: bool = True,
    EMA_20: bool = True,
    EMA_100: bool = True,
    EMA_150: bool = True,
    WMA_20: bool = True,
    WMA_100: bool = True,
    WMA_150: bool = True,
    Bollinger_Bands: bool = True
) -> Tuple[pd.DataFrame]:
    """
    Функция для расчета технических индикаторов и создания целевых переменных на основе предоставленного DataFrame.

    Параметры:
    - df (pd.DataFrame): Исходный DataFrame с данными. Ожидается, что он содержит колонки 'Close', 'Adj Close', 'Open'.
    - RSI (bool, по умолчанию True): Флаг для расчета индикатора относительной силы (RSI).
    - EMA_20 (bool, по умолчанию True): Флаг для расчета экспоненциальной скользящей средней (EMA) с периодом 20.
    - EMA_100 (bool, по умолчанию True): Флаг для расчета EMA с периодом 100.
    - EMA_150 (bool, по умолчанию True): Флаг для расчета EMA с периодом 150.
    - WMA_20 (bool, по умолчанию True): Флаг для расчета взвешенной скользящей средней (WMA) с периодом 20.
    - WMA_100 (bool, по умолчанию True): Флаг для расчета WMA с периодом 100.
    - WMA_150 (bool, по умолчанию True): Флаг для расчета WMA с периодом 150.
    - Bollinger_Bands (bool, по умолчанию True): Флаг для расчета полос Боллинджера.

    Возвращаемые значения:
    - pd.DataFrame: Измененный DataFrame с добавленными техническими индикаторами и целевыми переменными.

    Примечания:
    - Функция создает технические индикаторы в зависимости от переданных флагов и добавляет их в DataFrame.
    - Убедитесь, что в исходном DataFrame присутствуют необходимые колонки: 'Close', 'Adj Close', и 'Open'.
    """
    
    # Создание копии исходного DataFrame, чтобы не изменять исходный объект
    data = deepcopy(df)

    if RSI:
        data['RSI'] = ta.rsi(data['Close'], timeperiod=15)  # RSI
    if EMA_20:
        data['EMA_20'] = ta.ema(data['Close'], timeperiod=20)  # EMA 20
    if EMA_100:
        data['EMA_100'] = ta.ema(data['Close'], timeperiod=100)  # EMA 100
    if EMA_150:
        data['EMA_150'] = ta.ema(data['Close'], timeperiod=150)  # EMA 150
    if WMA_20:
        data['WMA_20'] = ta.wma(data['Close'], timeperiod=20)  # WMA 20
    if WMA_100:
        data['WMA_100'] = ta.wma(data['Close'], timeperiod=100)  # WMA 100
    if WMA_150:
        data['WMA_150'] = ta.wma(data['Close'], timeperiod=150)  # WMA 150
    if Bollinger_Bands:
        bands = ta.bbands(data['Close'], timeperiod=20)
        data['upper band'] = bands['BBU_5_2.0']  # Верхняя полоса
        data['middle band'] = bands['BBM_5_2.0']  # Средняя полоса
        data['lower band'] = bands['BBL_5_2.0']  # Нижняя полоса

    # Очистка данных 
    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    # тут пока не уверен нужно тестить 
    # data.drop(['Volume', 'Close', 'Date'], axis=1, inplace=True)
    # data.drop(['Date'], axis=1, inplace=True)

    return data
