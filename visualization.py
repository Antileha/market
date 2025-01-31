import matplotlib.pyplot as plt

def plot_predictions_with_future(actual, conv_t1, conv_t2, conv_t3, trans_t1, trans_t2, trans_t3):
    plt.figure(figsize=(12, 6))

    # ConvLSTM
    plt.plot(conv_t1, label="ConvLSTM RSI (t+1)", color='orange', alpha=0.7)
    plt.plot(conv_t2, label="ConvLSTM RSI (t+2)", color='red', linestyle="dashed", alpha=0.7)
    plt.plot(conv_t3, label="ConvLSTM RSI (t+3)", color='green', linestyle="dashed", alpha=0.7)

    # Transformer
    plt.plot(trans_t1, label="Transformer RSI (t+1)", color='purple', alpha=0.7)
    plt.plot(trans_t2, label="Transformer RSI (t+2)", color='pink', linestyle="dashed", alpha=0.7)
    plt.plot(trans_t3, label="Transformer RSI (t+3)", color='cyan', linestyle="dashed", alpha=0.7)

    plt.plot(actual, label="Фактические значения RSI", color='blue', alpha=0.7)
    plt.legend()
    plt.title("Сравнение ConvLSTM и Transformer в предсказании RSI")
    plt.xlabel("Временные шаги")
    plt.ylabel("RSI (нормализованный)")
    plt.grid(True)
    plt.show()
