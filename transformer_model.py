import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, output_size=1):
        super(TimeSeriesTransformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, output_size)
        self.embedding = nn.Linear(input_size, d_model)

    def forward(self, x):
        x = self.embedding(x)  # Преобразуем входные данные в пространство Transformer
        x = self.transformer_encoder(x)  # Пропускаем через Transformer
        return self.fc(x[:, -1, :])  # Берем последний шаг последовательности
