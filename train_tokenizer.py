import os
from tokenizer import BPETokenizer


def main():
    print("🔄 Обучаем токенизатор...")

    os.makedirs("models", exist_ok=True)
    tokenizer_path = "models/bpe_tokenizer.json"

    # Ищем доступный файл с текстом
    corpus_path = "resources/cultura_ru_subset.txt"
    if not os.path.exists(corpus_path):
        corpus_path = "resources/raw-wiktextract-data.jsonl"

    if not os.path.exists(corpus_path):
        print(f"❌ Не найден файл с текстом в resources/!")
        return

    # Передаем путь к файлу в зависимости от того, как устроен ваш BPETokenizer
    tokenizer = BPETokenizer()

    try:
        # Пробуем обучить, передавая путь к файлу списком (как требует Hugging Face tokenizers)
        tokenizer.train([corpus_path])
    except Exception as e:
        print(f"⚠️ Ошибка при передаче списком, пробуем передать строку: {e}")
        tokenizer.train(corpus_path)

    tokenizer.save(tokenizer_path)
    print(f"✅ Успешно! Токенизатор сохранен в `{tokenizer_path}`.")


if __name__ == "__main__":
    main()