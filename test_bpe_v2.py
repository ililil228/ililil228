from tokenizer import BPETokenizer

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

tokenizer = BPETokenizer(vocab_size=256)

print("Обучаем маленький тестовый BPE...")
tokenizer.tokenizer.train_from_iterator(
    texts,
    trainer=__import__("tokenizers").trainers.BpeTrainer(
        vocab_size=256,
        min_frequency=1,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        continuing_subword_prefix="##",
    )
)

print("\n" + "=" * 60)
print("TEST")
print("=" * 60)

for text in texts:
    encoded = tokenizer.tokenizer.encode(text)

    print("\nTEXT:")
    print(text)

    print("TOKENS:")
    print(encoded.tokens)

    print("IDS:")
    print(encoded.ids)

    decoded = tokenizer.tokenizer.decode(encoded.ids)

    print("DECODE:")
    print(decoded)

    print("MATCH:", text == decoded)