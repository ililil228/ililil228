from tokenizer import BPETokenizer

tokenizer = BPETokenizer()
tokenizer.load("models/bpe_tokenizer.json")

text = "Привет! Как дела?"

ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print("Исходный:")
print(text)

print("\nIDs:")
print(ids)

print("\nPieces:")
encoded = tokenizer.tokenizer.encode(text)

for token_id, piece in zip(encoded.ids, encoded.tokens):
    print(f"{token_id} -> {piece!r}")

print("\nDecode:")
print(decoded)

print("\nСовпадает:", text == decoded)

print("\nVocab:", tokenizer.get_vocab_size())