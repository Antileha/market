from data_loader import load_or_download_data
from indicators import add_indicators
from prepare_data import prepare_rsi_data
from hybrid_model import HybridTransformerConvLSTM
from optimize_hyperparams import optimize
from train_model import train_hybrid_model, save_model, load_model
from visualization import plot_predictions_with_future
import json

import os
import torch

if __name__ == "__main__":
    ticker = "EURUSD=X"
    model_path = "saved_models/hybrid_model.pth"
    params_path = "saved_models/best_params.json"

    # 1. 📥 Загрузка данных
    data, file_path = load_or_download_data(ticker)

    # 2. 📊 Добавление индикаторов
    data = add_indicators(data)

    # 3. 📈 Подготовка данных (прогноз на 1, 2 и 3 шага вперед)
    seq_length = 30
    X, y1, y2, y3, scaler = prepare_rsi_data(data, seq_length=seq_length)

    # 🔹 Разделение данных на train/test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y1_train, y1_test = y1[:train_size], y1[train_size:]

    # 4. 🧠 Проверяем, есть ли сохранённая модель
    if os.path.exists(model_path) and os.path.exists(params_path):
        print("📥 Загружаем сохранённую модель и гиперпараметры...")
        # Загружаем гиперпараметры
        with open(params_path, "r") as f:
            best_params = json.load(f)

        # Создаём модель с загруженными параметрами
        model = HybridTransformerConvLSTM(
            input_size=1, d_model=64, nhead=4, num_layers=best_params["num_layers"],
            conv_filters=best_params["conv_filters"], hidden_size=best_params["hidden_size"]
        )
        model = load_model(model, model_path)
    else:
        print("🚀 Оптимизируем гиперпараметры...")
        best_params = optimize(X_train, y1_train)

        # Создаём и обучаем модель
        model = HybridTransformerConvLSTM(
            input_size=1, d_model=64, nhead=4, num_layers=best_params["num_layers"],
            conv_filters=best_params["conv_filters"], hidden_size=best_params["hidden_size"]
        )
        model = train_hybrid_model(model, X_train, y1_train, num_epochs=200,
                                   learning_rate=best_params["learning_rate"], batch_size=best_params["batch_size"])

        params_path = "saved_models/best_params.json"

        # Проверяем, содержит ли best_params данные перед записью
        if not best_params:
            raise ValueError("Ошибка: best_params пуст, данные не были оптимизированы.")

        # Сохраняем гиперпараметры
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=4)

        print(f"✅ Гиперпараметры сохранены в {params_path}")

        # Сохраняем модель
        save_model(model, model_path)

    # 5. 🔍 Оценка модели
    model.eval()
    with torch.no_grad():
        predictions_t1 = model(X_test).numpy()
        predictions_t2 = model(X_test[:, 1:]).numpy()  # Для t+2
        predictions_t3 = model(X_test[:, 2:]).numpy()  # Для t+3
        actual = y1_test.numpy()

        # Обратное преобразование
        predictions_t1 = scaler.inverse_transform(predictions_t1)
        predictions_t2 = scaler.inverse_transform(predictions_t2)
        predictions_t3 = scaler.inverse_transform(predictions_t3)
        actual = scaler.inverse_transform(actual)

    # 6. 📊 Визуализация
    plot_predictions_with_future(actual, predictions_t1, predictions_t2, predictions_t3, predictions_t1, predictions_t2, predictions_t3)
