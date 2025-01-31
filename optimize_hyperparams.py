import torch
import optuna
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from hybrid_model import HybridTransformerConvLSTM

def train_hybrid_model(trial, X_train, y_train, num_epochs=30):  # 🔥 Уменьшаем num_epochs
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    conv_filters = trial.suggest_int("conv_filters", 16, 64)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    model = HybridTransformerConvLSTM(
        input_size=1, d_model=64, nhead=4, num_layers=num_layers, conv_filters=conv_filters, hidden_size=hidden_size
    )

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# Оптимизация с ускорением
def optimize(X_train, y_train):
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))  # 🔥 Ускоренные настройки
    study.optimize(lambda trial: train_hybrid_model(trial, X_train, y_train), n_trials=10, n_jobs=2)  # 🔥 Параллельный запуск и меньше итераций

    print("✅ Лучшие гиперпараметры:", study.best_params)
    return study.best_params
