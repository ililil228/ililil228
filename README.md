# 🤖 MY_AI V2 — Custom Russian Transformer-Decoder (PyTorch + DirectML)

<p align="left">
  <img src="https://github.com/user-attachments/assets/db221fa5-94d7-45a9-aab1-aa13633b6150" alt="Python" height="38" />&nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/9527ccaa-5498-4346-8a38-d8c9949e6cad" alt="PyTorch" height="38" />&nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/4883f368-31d3-4627-bf62-09dd5e838dfe" alt="AMD" height="38" />&nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/a3096208-befe-40c3-9e65-650a53d57225" alt="DirectX / DirectML" height="38" />&nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/bb346d18-ba09-4b6e-9603-fa8d94efa975" alt="Streamlit" height="38" />&nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/0af4281f-e4dd-464b-b0c5-d51c74e5037c" alt="Russian Language" height="38" />
</p>


> **Самодельный русскоязычный трансформер-декодер (11.56M параметров) на PyTorch, оптимизированный под видеокарты AMD через DirectML.**  
> *A custom Russian-only Decoder-only Transformer (11.56M parameters) built with PyTorch and optimized for AMD GPUs via DirectML.*

---

### ⚠️ Честный дисклеймер / Honest Disclaimer 🥒

* **🇷🇺 RU:** Эта нейросеть обучена **исключительно на русском языке** и на текущем этапе развития она **тупая как огурец 🥒**. Это не убийца ChatGPT, а честный R&D-эксперимент с нуля на маленькой архитектуре (~11.5M параметров). Главная цель проекта — отладить полный цикл (свой BPE-токенизатор, потоковую загрузку гигабайтных датасетов, микро-батчинг под VRAM и веб-интерфейс) на старой доброй видеокарте AMD RX 580.
* **🇬🇧 EN:** This model is trained **exclusively on the Russian language** and at this stage, it is **as dumb as a cucumber 🥒**. It is not meant to compete with ChatGPT—it is a pure R&D engineering sandbox (~11.5M parameters). The main goal is to build and debug the entire end-to-end LLM pipeline (custom BPE tokenizer, streaming multi-gigabyte datasets, gradient accumulation via micro-batching, and a Streamlit control panel) on an AMD RX 580 GPU.

---

## 🎥 Демонстрация работы / Live Demo
Обучение/learning 🎓
<img width="1888" height="897" alt="Work" src="https://github.com/user-attachments/assets/e904c2b6-efb5-48f0-9c7d-b12cd2bc2885" />


Ответ модели/ model's response 💌

<img width="896" height="757" alt="Answer" src="https://github.com/user-attachments/assets/cd0c78ea-262d-4be2-a5df-d3bfd4ec0864" />



---

## 🧠 Архитектура и характеристики / Model Specs

Модель построена по архитектуре **Decoder-only Transformer** (по принципу GPT) с кастомным BPE-токенизатором для русского языка:

* **Всего параметров (Total Parameters):** `11,564,544` (~11.56M)
* **Словарь (BPE Vocab Size):** `16,384` tokens (Russian-focused)
* **Размерность эмбеддингов (Embedding Dim):** `256`
* **Слои и головы (Layers / Attention Heads):** `4` layers / `4` heads
* **Скрытый слой FFN:** `256 → 1024 → 256`
* **Контекстное окно (Context Length):** `64` tokens
* **Оптимизация VRAM (Micro-batching):** `MICRO_BATCH_SIZE = 4` (позволяет крутить большие батчи без вылетов по памяти видеокарты)

---

## ⚙️ Тестовый стенд и стек / Hardware & Tech Stack

Проект разработан и протестирован на локальном «народном» железе с применением подхода **Vibe Coding / AI-Assisted Engineering** (быстрое прототипирование и отладка архитектуры с помощью LLM-агентов):

* **CPU:** AMD Ryzen 5 5600
* **GPU:** AMD Radeon RX 580 (accelerated via `torch-directml`)
* **RAM:** 16 GB DDR4 (потоковое чтение датасетов `Wiktextract` 2.35 GB и диалогов `842 MB` без забивания всей оперативки)
* **Frontend / Control Panel:** Streamlit (`app.py` — графики Loss в реальном времени, замер `шаг/мин`, ETA, чат-инференс и диагностика чекпоинтов)
* **Quick Launch:** Интерактивный `start.bat` для Windows

---

## 🚀 Быстрый запуск (Windows) / Quick Start

1. Клонируйте репозиторий / Clone the repo:
   ```bash
   git clone https://github.com
   cd ТВОЙ_РЕПОЗИТОРИЙ
   ```

2. Создайте виртуальное окружение и установите зависимости / Set up venv & install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Запустите панель управления через `start.bat` или напрямую / Run via `start.bat` or Streamlit:
   ```bash
   streamlit run app.py
   ```

---

## 🗺️ План развития / Roadmap

- [x] Сборка кастомного русского BPE-токенизатора (`16384` токенов)
- [x] Реализация архитектуры Transformer-Decoder (11.56M) на PyTorch
- [x] Адаптация обучения и инференса под AMD RX 580 через DirectML
- [x] Потоковый парсер крупных датасетов (`.jsonl`) и веб-интерфейс на Streamlit
- [ ] **Linux Port (Ubuntu Server):** Адаптация проекта под Linux (CPU / ROCm / Vulkan) для запуска в headless-режиме на домашнем сервере
- [ ] Увеличение контекста и дообучение модели, чтобы она стала хотя бы немного умнее огурца 🥒
