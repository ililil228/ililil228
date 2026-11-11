import os

from tokenizer import BPETokenizer
from jsonl_streamer import stream_wiktextract_text


# ============================================================
# CONFIG
# ============================================================

VOCAB_SIZE = 16384

OUTPUT_PATH = "models/bpe_tokenizer_v2.json"

WIKTEXTRACT_PATH = "resources/raw-wiktextract-data.jsonl"
RUSSIAN_WORDS_PATH = "resources/100000-russian-words.txt"

MAX_WIKTEXTRACT_SAMPLES = 500_000
MAX_RUSSIAN_WORDS = 100_000
MAX_CULTURA_SAMPLES = 100_000


# ============================================================
# LOCAL DATA
# ============================================================

def stream_wiktextract():
    if not os.path.exists(WIKTEXTRACT_PATH):
        raise FileNotFoundError(
            f"Не найден файл:\n{WIKTEXTRACT_PATH}"
        )

    print("Reading Wiktextract...")

    count = 0

    for text in stream_wiktextract_text(
        WIKTEXTRACT_PATH,
        max_lines=MAX_WIKTEXTRACT_SAMPLES
    ):
        if not text:
            continue

        text = text.strip()

        if not text:
            continue

        yield text

        count += 1

        if count % 100_000 == 0:
            print(f"Wiktextract samples: {count:,}")

    print(f"Wiktextract samples used: {count:,}")


def stream_russian_words():
    if not os.path.exists(RUSSIAN_WORDS_PATH):
        print(
            f"WARNING: file not found: "
            f"{RUSSIAN_WORDS_PATH}"
        )
        return

    print("Reading Russian words...")

    count = 0

    with open(
        RUSSIAN_WORDS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            word = line.strip()

            if not word:
                continue

            yield word

            count += 1

            if count >= MAX_RUSSIAN_WORDS:
                break

    print(f"Russian words used: {count:,}")


# ============================================================
# CULTURA
# ============================================================

def stream_cultura():
    """
    Безопасный streaming-проход Cultura.

    Если Hugging Face долго не отдаёт данные,
    этот генератор не имеет смысла бесконечно ждать.
    """

    from datasets import load_dataset

    print()
    print("=" * 60)
    print("LOADING CULTURA")
    print("=" * 60)

    print("Connecting to Hugging Face...")

    try:
        dataset = load_dataset(
            "deepvk/cultura_ru_edu",
            split="train",
            streaming=True
        )
    except Exception as e:
        print()
        print("WARNING: Cultura не загрузилась.")
        print(f"Причина: {e}")
        print("Продолжаем без Cultura.")
        return

    print("Cultura dataset connected.")
    print("Reading samples...")

    count = 0

    try:
        for item in dataset:

            if count >= MAX_CULTURA_SAMPLES:
                break

            text = None

            if isinstance(item, dict):
                text = item.get("text")

                if text is None:
                    parts = []

                    for value in item.values():
                        if isinstance(value, str):
                            parts.append(value)

                    text = " ".join(parts)

            if not text:
                continue

            text = text.strip()

            if not text:
                continue

            yield text

            count += 1

            if count % 10_000 == 0:
                print(
                    f"Cultura samples: "
                    f"{count:,}/{MAX_CULTURA_SAMPLES:,}"
                )

    except Exception as e:
        print()
        print("WARNING: ошибка чтения Cultura.")
        print(f"Причина: {e}")
        print("Останавливаем Cultura и продолжаем.")


    print(
        f"Cultura samples used: {count:,}"
    )


# ============================================================
# COMBINED STREAM
# ============================================================

def combined_stream():

    print()
    print("=" * 60)
    print("BUILDING TRAINING STREAM")
    print("=" * 60)

    # 1. Локальный Wiktextract
    yield from stream_wiktextract()

    # 2. Локальный список русских слов
    yield from stream_russian_words()

    # 3. Cultura
    yield from stream_cultura()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MY_AI — BPE TOKENIZER V2")
    print("=" * 60)

    print()
    print("Configuration:")
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Output:     {OUTPUT_PATH}")

    print()
    print("Datasets:")
    print(f"- Wiktextract: {WIKTEXTRACT_PATH}")
    print(f"- Russian words: {RUSSIAN_WORDS_PATH}")
    print("- deepvk/cultura_ru_edu")

    print()

    if os.path.exists(OUTPUT_PATH):

        print(
            f"WARNING: {OUTPUT_PATH} already exists."
        )

        answer = input(
            "Перезаписать? [y/N]: "
        ).strip().lower()

        if answer != "y":
            print("Отменено.")
            return

    tokenizer = BPETokenizer(
        vocab_size=VOCAB_SIZE
    )

    print()
    print("=" * 60)
    print("START BPE TRAINING")
    print("=" * 60)

    try:

        tokenizer.train_from_iterator(
            combined_stream(),
            save_path=OUTPUT_PATH
        )

    except KeyboardInterrupt:

        print()
        print("Остановлено пользователем.")

        return

    except Exception as e:

        print()
        print("=" * 60)
        print("ERROR")
        print("=" * 60)

        print(e)

        return

    print()
    print("=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    tokenizer.load(OUTPUT_PATH)

    print()
    print("Running tokenizer check...")

    tokenizer.check()

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(f"Tokenizer: {OUTPUT_PATH}")
    print(
        f"Vocab: {tokenizer.get_vocab_size()}"
    )

    print()
    print("Старый tokenizer НЕ изменён.")
    print("Старый checkpoint НЕ изменён.")

    print()
    print(
        "Следующий этап: "
        "новый Transformer с V2 tokenizer."
    )


if __name__ == "__main__":
    main()