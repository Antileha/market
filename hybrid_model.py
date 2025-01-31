import torch
import torch.nn as nn

class HybridTransformerConvLSTM(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, conv_filters, hidden_size):
        super(HybridTransformerConvLSTM, self).__init__()

        # 📌 Используем conv_filters как input_size для LSTM
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv_filters, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=conv_filters, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Предсказание одного значения

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 1)  # Транспонируем для LSTM
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Выбираем последний шаг
        return x

    def to_device(self):
        """Перемещает модель на доступное устройство"""
        self.to(self.device)
