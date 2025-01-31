import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from indicators import add_indicators

def prepare_test_data(data, seq_length=30):
    """
    Подготавливает тестовые данные для предсказания RSI.
    """
    # 📊 Добавляем индикаторы
    data = add_indicators(data)

    # 🏷 Берём только RSI 21
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['RSI_21']])

    # 📉 Создаём последовательности
    sequences = []
    for i in range(len(scaled_data) - seq_length):
        sequences.append(scaled_data[i:i + seq_length])

    return np.array(sequences), scaler
