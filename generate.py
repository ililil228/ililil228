import torch
import torch.nn as nn
import torch_directml

from tokenizer import BPETokenizer
from model import BPEModel  # Импортируем исправленный трансформер

# ==========================================
# КОНФИГУРАЦИЯ ОБНОВЛЕННОЙ АРХИТЕКТУРЫ
# ==========================================
# Эти параметры ДОЛЖНЫ совпадать с теми, на которых обучалась модель
EMBED_DIM = 256  # Увеличенный размер эмбеддинга (было 64)
NUM_HEADS = 8  # Число голов внимания (должно нацело делить EMBED_DIM: 256 / 8 = 32)
NUM_LAYERS = 6  # Глубина сети (число слоев трансформера)
MAX_SEQ_LEN = 64  # Максимальная длина контекста модели

TOKENIZER_PATH = "models/bpe_tokenizer_v2.json"
CHECKPOINT_PATH = "models/bpe_model_v2.pt"

# Автоматический выбор устройства (DirectML для AMD, иначе CPU)
device = torch_directml.device() if hasattr(torch_directml, 'device') else torch.device('cpu')

# ==========================================
# ИНИЦИАЛИЗАЦИЯ ТОКЕНИЗАТОРА
# ==========================================
tokenizer = BPETokenizer()
tokenizer.load(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()


# ==========================================
# ФУНКЦИЯ ГЕНЕРАЦИИ ТЕКСТА НА ТРАНСФОРМЕРЕ
# ==========================================
def generate_text(model, tokenizer, start_str, gen_length=50, temperature=0.7, top_k=50):
    model.eval()

    # Кодируем стартовый промпт с помощью BPE
    input_ids = tokenizer.encode(start_str)

    if not input_ids:
        return "Ошибка: промпт не удалось закодировать!"

    # Ограничиваем начальный контекст максимальным окном трансформера
    input_ids = input_ids[-MAX_SEQ_LEN:]

    # Создаем тензор структуры [Batch=1, Seq_len] и отправляем на GPU
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(gen_length):
            # Трансформер принимает строго фиксированное окно контекста
            x_cond = x[:, -MAX_SEQ_LEN:]

            # Прямой проход модели (возвращает logits и None)
            logits, _ = model(x_cond)

            # Извлекаем логиты только для ПОСЛЕДНЕГО сгенерированного токена
            logits = logits[:, -1, :]
            logits = logits / temperature
            logits = logits.squeeze(0)  # Приводим к размеру [vocab_size]

            # Фильтрация Top-K, отсекающая низковероятный текстовый коллапс
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float('Inf')

            # Расчет вероятностей через Softmax и сэмплирование нового токена
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Конкатенируем новый токен по оси времени к текущей последовательности
            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

    # Переносим полученные ID токенов обратно на CPU и декодируем их в текст
    generated_ids = x.detach().cpu().tolist()[0]
    return tokenizer.decode(generated_ids)


# ==========================================
# ОСНОВНОЙ ЗАПУСК
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING TRANSFORMER GENERATION (BPEModel)")
    print("=" * 60)
    print("Device:", device)
    print("Tokenizer vocab size:", vocab_size)
    print()

    # Создаем и инициализируем трансформер вместо старой GRU
    model = BPEModel(
        vocab_size=vocab_size,  # Динамически берется из подгруженного токенизатора (16384 или 32000)
        embed_size=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)

    try:
        # Загружаем веса с маппингом на CPU перед отправкой на DirectML (защита от крашей драйверов AMD)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

        # Проверяем структуру сохранения (словарь состояния или чистые веса)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Веса успешно загружены! Шаг чекпоинта: {checkpoint.get('step', '?')}")
        else:
            model.load_state_dict(checkpoint)
            print("Веса успешно загружены!")

    except FileNotFoundError:
        print(f"Внимание: файл весов '{CHECKPOINT_PATH}' не найден. Модель генерирует случайный текст.")

    prompt = "Егор"
    print(f"\n--- Результат генерации по промпту '{prompt}': ---")

    # Тестируем генерацию с температурой 0.75 и Top-K=40
    text_result = generate_text(
        model,
        tokenizer,
        start_str=prompt,
        gen_length=40,
        temperature=0.75,
        top_k=40
    )
    print(text_result)
    print("\n" + "=" * 60)
