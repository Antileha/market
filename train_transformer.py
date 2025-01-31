import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

def train_transformer_model(model, X_train, y_train, num_epochs=200, learning_rate=0.0003, batch_size=32):
    """
    Обучение модели Transformer.
    """
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

        if (epoch + 1) % 10 == 0:
            print(f"🔄 Эпоха [{epoch+1}/{num_epochs}], Средний Loss: {epoch_loss / len(dataloader):.6f}")

    return model
