import pandas_ta as ta

def add_indicators(data):
    """
    Добавляет индикаторы RSI и EMA в данные.
    """
    # Добавляем RSI
    data['RSI_14'] = ta.rsi(data['Close'], length=14)
    data['RSI_21'] = ta.rsi(data['Close'], length=21)
    data['RSI_50'] = ta.rsi(data['Close'], length=50)

    # Добавляем EMA
    data['EMA_13'] = ta.ema(data['Close'], length=13)
    data['EMA_50'] = ta.ema(data['Close'], length=50)

    # Добавляем разницу между EMA_13 и Close
    data['EMA_13_Close_Diff'] = data['EMA_13'] - data['Close']

    return data
