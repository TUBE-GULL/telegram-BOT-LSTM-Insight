from copy import deepcopy
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preparing_data_for_lstm(df, features, target, seq_length):
    """
    Функция для подготовки данных для обучения модели LSTM.

    Параметры:
    - df (pd.DataFrame): Исходный DataFrame с данными. Ожидается, что он содержит колонки, указанные в параметрах features и target, а также колонку 'Date'.
    - features (list): Список названий колонок, которые будут использоваться в качестве входных признаков для модели LSTM.
    - target (str): Название колонки, которая будет использоваться в качестве целевой переменной.
    - seq_length (int): Длина временного окна, определяющая количество временных шагов, которые будут использоваться для создания входных данных для модели LSTM.

    Возвращаемые значения:
    - X_train (np.ndarray): Массив входных данных для обучения модели LSTM.
    - X_test (np.ndarray): Массив входных данных для тестирования модели LSTM.
    - y_train (np.ndarray): Массив целевых значений для обучения модели LSTM.
    - y_test (np.ndarray): Массив целевых значений для тестирования модели LSTM.

    Описание:
    1. **Создание копии DataFrame:** Создается копия исходного DataFrame для предотвращения изменений в оригинальных данных.
    2. **Подготовка данных:**
       - Преобразование колонки 'Date' в формат datetime и установка её в качестве индекса DataFrame.
       - Нормализация данных с использованием MinMaxScaler, чтобы привести значения в диапазон от 0 до 1. Нормализация применяется к указанным признакам и целевой переменной.
    3. **Создание временных окон:**
       - Функция `create_sequences` создает временные окна (последовательности) для входных данных и соответствующих целевых значений. Входные данные формируются из предыдущих `seq_length` шагов, а целевые значения берутся из следующего шага после временного окна.
    4. **Разделение данных:**
       - Данные разделяются на обучающую и тестовую выборки с использованием `train_test_split`. Важно, что данные разделяются без перемешивания (shuffle=False), чтобы сохранить временную последовательность.

    Примечания:
    - Убедитесь, что в исходном DataFrame присутствуют все указанные колонки: 'Date', а также колонки, указанные в `features` и `target`.
    - Функция полезна для подготовки данных для обучения моделей LSTM, которые требуют входных данных в виде временных окон.
    - Нормализация данных помогает улучшить сходимость и производительность модели LSTM.
    """
    
    # Создание копии исходного DataFrame, чтобы не изменять исходный объект
    data = deepcopy(df)

    # Подготовка данных
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Нормализация данных
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features + [target]])

    # Функция для создания временных окон
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length, :-1]
            y = data[i+seq_length, -1]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X, y = create_sequences(scaled_data, seq_length)

    # Разделение данных на обучающие и тестовые выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    return X_train, X_test, y_train, y_test
 
 
# def preparing_data_for_lstm(df, features, target, seq_length):
#     """
#     Функция для подготовки данных для обучения модели LSTM.

#     Параметры:
#     - df (pd.DataFrame): Исходный DataFrame с данными. Ожидается, что он содержит колонки, указанные в параметрах features и target, а также колонку 'Date'.
#     - features (list): Список названий колонок, которые будут использоваться в качестве входных признаков для модели LSTM.
#     - target (str): Название колонки, которая будет использоваться в качестве целевой переменной.
#     - seq_length (int): Длина временного окна, определяющая количество временных шагов, которые будут использоваться для создания входных данных для модели LSTM.

#     Возвращаемые значения:
#     - X_train (np.ndarray): Массив входных данных для обучения модели LSTM.
#     - X_test (np.ndarray): Массив входных данных для тестирования модели LSTM.
#     - y_train (np.ndarray): Массив целевых значений для обучения модели LSTM.
#     - y_test (np.ndarray): Массив целевых значений для тестирования модели LSTM.

#     Описание:
#     1. **Создание копии DataFrame:** Создается копия исходного DataFrame для предотвращения изменений в оригинальных данных.
#     2. **Подготовка данных:**
#        - Преобразование колонки 'Date' в формат datetime и установка её в качестве индекса DataFrame.
#        - Нормализация данных с использованием MinMaxScaler, чтобы привести значения в диапазон от 0 до 1. Нормализация применяется к указанным признакам и целевой переменной.
#     3. **Создание временных окон:**
#        - Функция `create_sequences` создает временные окна (последовательности) для входных данных и соответствующих целевых значений. Входные данные формируются из предыдущих `seq_length` шагов, а целевые значения берутся из следующего шага после временного окна.
#     4. **Разделение данных:**
#        - Данные разделяются на обучающую и тестовую выборки с использованием `train_test_split`. Важно, что данные разделяются без перемешивания (shuffle=False), чтобы сохранить временную последовательность.

#     Примечания:
#     - Убедитесь, что в исходном DataFrame присутствуют все указанные колонки: 'Date', а также колонки, указанные в `features` и `target`.
#     - Функция полезна для подготовки данных для обучения моделей LSTM, которые требуют входных данных в виде временных окон.
#     - Нормализация данных помогает улучшить сходимость и производительность модели LSTM.
#     """
    
#     # Создание копии исходного DataFrame, чтобы не изменять исходный объект
#     data = deepcopy(df)

#     # Подготовка данных
#     data['Date'] = pd.to_datetime(data['Date'])
#     data.set_index('Date', inplace=True)

#     # Нормализация данных
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(data[features + [target]])

#     # Функция для создания временных окон
#     def create_sequences(data, seq_length):
#         xs, ys = [], []
#         for i in range(len(data) - seq_length):
#             x = data[i:i+seq_length, :-1]
#             y = data[i+seq_length, -1]
#             xs.append(x)
#             ys.append(y)
#         return np.array(xs), np.array(ys)

#     X, y = create_sequences(scaled_data, seq_length)

#     # Разделение данных на обучающие и тестовые выборки
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
#     return X_train, X_test, y_train, y_test
 