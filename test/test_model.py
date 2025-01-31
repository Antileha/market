import os
import json
import torch
import sys
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prepare_test_data import prepare_test_data
from data_loader import load_or_download_data
from train_model import load_model
from hybrid_model import HybridTransformerConvLSTM

# 📌 Проверка даты
def validate_date(date_str, default):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        print(f"⚠ Ошибка! Некорректный формат даты: {date_str}. Используется {default}.")
        return default

# 🔹 Парсинг аргументов
parser = argparse.ArgumentParser(description="Тестирование модели на новых данных")
parser.add_argument("--ticker", type=str, default="EURUSD=X", help="Тикер (например, AAPL, BTC-USD)")
parser.add_argument("--start_date", type=str, default="2015-01-01", help="Дата начала (YYYY-MM-DD)")
parser.add_argument("--end_date", type=str, default="2025-01-28", help="Дата окончания (YYYY-MM-DD)")
args = parser.parse_args()

# 🔄 1. Загрузка данных
print(f"🔍 Загружаем данные для {args.ticker} ({args.start_date} - {args.end_date})")
data, file_path = load_or_download_data(args.ticker, args.start_date, args.end_date)

# 2. 📊 Подготовка данных
seq_length = 30
X_test, scaler = prepare_test_data(data, seq_length=seq_length)

# 3. 📥 Загрузка гиперпараметров
params_path = os.path.join("saved_models", "best_params.json")
if not os.path.exists(params_path):
    raise FileNotFoundError(f"❌ Файл {params_path} не найден!")

with open(params_path, "r") as f:
    best_params = json.load(f)

# 4. 🧠 Загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HybridTransformerConvLSTM(
    input_size=1, d_model=64, nhead=4, num_layers=best_params["num_layers"],
    conv_filters=best_params["conv_filters"], hidden_size=best_params["hidden_size"]
).to(device)

model = load_model(model, "saved_models/hybrid_model.pth")
model.eval()

# ✅ Подгоняем входные данные
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# ✅ Используем правильный формат для Conv1D
X_test_tensor = X_test_tensor.view(X_test_tensor.shape[0], 1, seq_length)  # (batch_size, 1, seq_length)

# 🔍 Вывод отладочной информации
print(f"✅ Итоговый размер X_test_tensor перед Conv1D: {X_test_tensor.shape}")

# 🔍 **Предсказания**
num_days = 30  # Количество последних дней для тестирования
X_test_last_month = X_test[-num_days:]  # Берем последние 30 дней
dates_last_month = data.index[-num_days:]  # Даты для теста
actual_rsi = data["RSI_21"].values[-num_days:]  # Фактические RSI

predictions_t1_list, predictions_t2_list, predictions_t3_list = [], [], []

# 🔍 **Используем модель правильно**
with torch.no_grad():
    for i in range(num_days):
        X_test_single = X_test_last_month[i].reshape(1, 1, seq_length)  # (batch, channels, seq_length)
        X_test_tensor = torch.tensor(X_test_single, dtype=torch.float32).to(device)

        # ✅ Корректный проход через модель
        x = model.conv1(X_test_tensor).permute(0, 2, 1)  # Перестановка размерностей для LSTM
        predictions, (hn, cn) = model.lstm(x)  # LSTM с сохранением скрытых состояний

        # ✅ Получаем предсказания для t+1, t+2, t+3
        predictions_t1 = predictions[:, -1, 0].cpu().numpy().flatten()
        predictions_t2 = predictions[:, -1, 1].cpu().numpy().flatten()
        predictions_t3 = predictions[:, -1, 2].cpu().numpy().flatten()

        # 🔄 Обратное преобразование
        predictions_t1 = scaler.inverse_transform(predictions_t1.reshape(-1, 1)).flatten()[0]
        predictions_t2 = scaler.inverse_transform(predictions_t2.reshape(-1, 1)).flatten()[0]
        predictions_t3 = scaler.inverse_transform(predictions_t3.reshape(-1, 1)).flatten()[0]

        # ✅ **Добавляем предсказания для текущего дня**
        predictions_t1_list.append(predictions_t1)
        predictions_t2_list.append(predictions_t2)
        predictions_t3_list.append(predictions_t3)

# 📊 **Формируем таблицу результатов**
df_results = pd.DataFrame({
    "Дата": dates_last_month,
    "Фактическое RSI": actual_rsi,
    "Прогноз RSI (t+1)": predictions_t1_list,
    "Прогноз RSI (t+2)": predictions_t2_list,
    "Прогноз RSI (t+3)": predictions_t3_list
})

# 📁 **Сохранение в CSV**
output_file = "test_results_last_month.csv"
df_results.to_csv(output_file, index=False)
print(f"✅ Результаты сохранены в {output_file}")

# 📊 **Вывод таблицы**
import ace_tools as tools
tools.display_dataframe_to_user(name="Результаты тестирования за последний месяц", dataframe=df_results)

print("✅ Тестирование завершено!")
