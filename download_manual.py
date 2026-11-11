import urllib.request
import os

print("🔄 Скачиваем пример текстов для обучения...")
os.makedirs("resources", exist_ok=True)
output_path = "resources/cultura_ru_subset.txt"

# Скачиваем небольшой открытый корпус текстов на русском языке для теста
url = "https://raw.githubusercontent.com/HarshCasper/Russian-Text-Corpus/main/test.txt" # или любой другой стабильный URL с текстом

try:
    urllib.request.urlretrieve(url, output_path)
    print(f"✅ Успешно скачано! Файл сохранен как `{output_path}`.")
except Exception as e:
    print(f"⚠️ Не удалось скачать по ссылке: {e}")
    print("💡 Вы можете просто положить любой текстовый файл с русскими книгами или статьями в папку `resources/` и переименовать в `cultura_ru_subset.txt`.")