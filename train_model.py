import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import os

def save_model(model, filename="saved_models/hybrid_model.pth"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(model.state_dict(), filename)
    print(f"✅ Модель сохранена в {filename}")

def load_model(model, filename="saved_models/hybrid_model.pth"):
    checkpoint = torch.load(filename)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    print(f"✅ Модель загружена из {filename} (только совпадающие параметры)")
    return model

def train_hybrid_model(model, X_train, y_train, num_epochs=200, learning_rate=0.0003, batch_size=32):
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

    save_model(model, "saved_models/hybrid_model.pth")
    return model
