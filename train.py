import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from model import LSTMModel
from data_loader import load_or_download_data
from data_preprocessor import prepare_data


def train_and_save_model(ticker, save_path="model.pth"):
    # Загрузка и подготовка данных
    data = load_or_download_data(ticker)
    X, y, scaler = prepare_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    # Определение модели
    input_size = 1
    hidden_size = 50
    num_layers = 2
    output_size = 1
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # Функция потерь и оптимизатор
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Обучение
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Эпоха [{epoch + 1}/{num_epochs}], Потери: {loss.item():.4f}")

    # Сохранение модели
    torch.save(model.state_dict(), save_path)
    print(f"Модель сохранена в {save_path}")

    return model, X_test, y_test, scaler
