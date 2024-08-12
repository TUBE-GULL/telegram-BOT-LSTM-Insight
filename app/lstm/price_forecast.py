import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from app.lstm.add_indicator_df import add_indicator_df
from app.lstm.add_target_df import add_target_df
# from preparing_data_for_lstm import preparing_data_for_lstm


def price_forecast(model,name_df,date): 
    """
    Прогнозирование цен на основе модели LSTM.

    Параметры:
    - model (str): Тип модели для загрузки ('shares', 'index', 'crypto').
    - name_df (str): Тикер актива для получения данных.
    - date (str): Конечная дата для получения данных.

    Возвращаемые значения:
    - data (pd.DataFrame): Исходные данные с датами и закрывающими ценами.
    - results_df (pd.DataFrame): DataFrame с предсказанными ценами и датами.
    """
    
    # Путь к сохраненной модели
    if model == 'shares':
        model_path = 'app/lstm/model_shares.keras'
    elif model == 'index':
        model_path = 'app/lstm/model_index.keras'
    elif model == 'crypto':
        model_path = 'app/lstm/model_crypto.keras'
    else:
        model_path = 'app/lstm/model_index.keras'
            
    # Загрузка модели
    model = load_model(model_path)

    # Определите размер временного окна (например, 10 дней)
    SEQ_LENGTH = 10

    # Выберите признаки и цель
    features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'RSI', 'EMA_20', 'EMA_100', 'EMA_150', 'WMA_20', 'WMA_100', 'WMA_150']
    target = 'TargetNextClose'

    data_spx = yf.download(tickers=name_df, start='1999-01-01', end=date)# Получение данных о S&P 500
    data = add_indicator_df(add_target_df(data_spx))

    # Нормализуйте данные
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features + [target]])

    # Подготовка новых данных для прогноза
    def prepare_input(data, seq_length):
        last_sequence = data[-seq_length:]
        return last_sequence.reshape((1, seq_length, len(features)))

    # Предположим, что у вас есть данные в переменной `new_data`
    new_data = scaled_data[:, :-1]  # Только признаки, без целевой переменной

    # Создайте входные данные для прогноза
    input_sequence = prepare_input(new_data, SEQ_LENGTH)

    # Прогноз на несколько дней вперед
    def predict_multiple_days(model, input_sequence, days, scaler):
        predictions = []
        for _ in range(days):
            predicted_scaled = model.predict(input_sequence)
            
            # Сохраните предсказанное значение
            predictions.append(predicted_scaled[0, 0])
            
            # Обновите входные данные
            new_input = np.copy(input_sequence)
            # Добавьте новое предсказанное значение к последнему окну
            predicted_scaled_reshaped = np.reshape(predicted_scaled, (1, 1, 1))  # размерность (1, 1, 1)
            # Обновите входные данные
            new_input = np.concatenate([new_input[:, 1:, :], np.tile(predicted_scaled_reshaped, (1, 1, new_input.shape[2]))], axis=1)
            input_sequence = new_input
        
        # Денормализуйте предсказания
        scaled_target = np.zeros((days, len(features) + 1))
        scaled_target[:, -1] = predictions
        denorm_target = scaler.inverse_transform(scaled_target)
        
        return denorm_target[:, -1]

    # Прогноз на 7 дней вперед
    predicted_values = predict_multiple_days(model, input_sequence, days=7, scaler=scaler)
    print(f'Predicted Next 7 Days Close Prices: {predicted_values}')

    # Последняя дата из оригинального DataFrame
    last_date = data['Date'].iloc[-1]
    # Количество предсказанных значений
    num_predictions = len(predicted_values)
    # Генерация новых дат, начиная с последней даты
    new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_predictions, freq='D')
    # Если нужно, вы можете создать DataFrame с результатами
    results_df = pd.DataFrame({
        'Predicted_Close': predicted_values.flatten(),
        'Date': new_dates,
    })

    data[['Date','Close']]
    # print(results_df) #    Predicted_Close       Date
    # print(data.head())
    results_df['Date'] = pd.to_datetime(results_df['Date'])
    
    return data[['Date','Close']] , results_df

# Построение графика
# plt.figure(figsize=(14, 7))
# plt.plot(data['Date'], data['Close'], label='Actual Close', marker='o')
# plt.plot(results_df['Date'], results_df['Predicted_Close'], label='Predicted Close', linestyle='--', marker='x')

# plt.xlabel('Date')
# plt.ylabel('Close Price')
# plt.title('Actual vs Predicted Close Prices')
# plt.legend()
# plt.grid(True)
# plt.xticks(rotation=45)
# plt.tight_layout()

# plt.show()