import os
import pandas as pd
import pandas_ta as ta


def add_missing_indicators(file_path):
    """
    Добавляет недостающие индикаторы RSI и EMA в файл данных.
    """
    # Проверяем, существует ли файл
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")

    # Чтение данных из файла
    data = pd.read_csv(file_path, index_col="Date", parse_dates=True)

    # Проверяем и добавляем недостающие индикаторы
    indicators_added = False

    if 'RSI_50' not in data.columns:
        data['RSI_50'] = ta.rsi(data['Close'], length=50)
        indicators_added = True
    if 'RSI_21' not in data.columns:
        data['RSI_21'] = ta.rsi(data['Close'], length=21)
        indicators_added = True
    if 'RSI_14' not in data.columns:
        data['RSI_14'] = ta.rsi(data['Close'], length=14)
        indicators_added = True
    if 'EMA_13' not in data.columns:
        data['EMA_13'] = ta.ema(data['Close'], length=13)
        indicators_added = True
    if 'EMA_50' not in data.columns:
        data['EMA_50'] = ta.ema(data['Close'], length=50)
        indicators_added = True

    # Если индикаторы добавлены, перезаписываем файл
    if indicators_added:
        data.to_csv(file_path)
        print(f"Индикаторы добавлены и файл {file_path} обновлен.")
    else:
        print(f"Все индикаторы уже присутствуют в файле {file_path}.")

    return data
