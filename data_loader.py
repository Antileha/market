import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def load_or_download_data(ticker, start_date="2015-01-01", end_date="2024-01-01"):
    """
    Загружает данные из локального файла или скачивает с Yahoo Finance, дописывая недостающие периоды.
    """
    file_path = f"data/{ticker}.csv"
    need_download = True  # Флаг необходимости скачивания

    # ✅ Проверяем, есть ли уже данные на диске
    if os.path.exists(file_path):
        print(f"✅ Найден локальный файл {file_path}, загружаем...")
        data = pd.read_csv(file_path, index_col="Date", parse_dates=True)

        # ✅ Проверяем, покрывает ли файл запрошенные даты
        file_start = data.index.min().strftime("%Y-%m-%d")
        file_end = data.index.max().strftime("%Y-%m-%d")

        if file_start <= start_date and file_end >= end_date:
            print(f"✅ Данные полностью покрывают диапазон {start_date} - {end_date}.")
            need_download = False
        else:
            print(f"🔄 Данные в файле с {file_start} по {file_end}, дополняем недостающие периоды...")
            need_download = True  # Нужно скачать недостающие данные

    if need_download:
        print(f"📥 Скачивание недостающих данных для {ticker} ({start_date} - {end_date})...")
        new_data = yf.download(ticker, start=start_date, end=end_date)

        if os.path.exists(file_path):
            # 📌 Объединяем старые и новые данные
            data = pd.concat([data, new_data]).sort_index().drop_duplicates()
        else:
            data = new_data

        # 💾 Сохраняем актуализированные данные
        data.to_csv(file_path)
        print(f"✅ Данные обновлены и сохранены в {file_path}")

    return data, file_path
