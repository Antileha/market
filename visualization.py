import matplotlib.pyplot as plt
import torch


def plot_predictions(model, X_test, y_test):
    """
    Визуализация прогнозов модели.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(y_test.numpy(), label="Фактические значения")
    plt.plot(predictions, label="Прогнозируемые значения")
    plt.legend()
    plt.title("Прогноз RSI")
    plt.show()
