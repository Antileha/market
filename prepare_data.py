import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def prepare_rsi_data(data, seq_length=30):
    """
    Подготавливает данные для прогнозирования RSI на 1, 2 и 3 шага вперед.
    """
    rsi_data = data[['RSI_21']].dropna()

    # Нормализация данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_rsi = scaler.fit_transform(rsi_data)

    # Создание последовательностей
    sequences, labels_t1, labels_t2, labels_t3 = [], [], [], []
    for i in range(len(scaled_rsi) - seq_length - 3):  # -3 для предсказания на 3 шага
        sequences.append(scaled_rsi[i:i + seq_length])  # Входные данные (30 шагов)
        labels_t1.append(scaled_rsi[i + seq_length])    # Прогноз RSI (t+1)
        labels_t2.append(scaled_rsi[i + seq_length + 1])  # Прогноз RSI (t+2)
        labels_t3.append(scaled_rsi[i + seq_length + 2])  # Прогноз RSI (t+3)

    # Преобразуем в тензоры PyTorch
    X = torch.tensor(np.array(sequences), dtype=torch.float32)
    y1 = torch.tensor(np.array(labels_t1), dtype=torch.float32)
    y2 = torch.tensor(np.array(labels_t2), dtype=torch.float32)
    y3 = torch.tensor(np.array(labels_t3), dtype=torch.float32)

    return X, y1, y2, y3, scaler
