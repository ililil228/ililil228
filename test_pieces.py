from tokenizer import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.load("models/bpe_tokenizer.json")

texts = [
    "Привет! Как дела?",
    "Я изучаю русский язык.",
    "Искусственный интеллект изучает русский язык.",
    "Трансформер использует механизм внимания.",
    "Москва — столица России.",
    "Это простой тест.",
    "Мой компьютер работает на Windows.",
    "Сегодня хорошая погода.",
]

for text in texts:
    encoded = tokenizer.tokenizer.encode(text)

    print("=" * 70)
    print("TEXT:")
    print(text)

    print("\nPIECES:")
    print(" | ".join(repr(x) for x in encoded.tokens))

    print("\nIDS:")
    print(encoded.ids)

    print("\nDECODE:")
    print(tokenizer.decode(encoded.ids))