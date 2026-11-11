import torch
import torch.nn as nn
from tokenizer import BPETokenizer

SEQ_LEN = 32
EMBED_DIM = 64
HIDDEN_DIM = 128

# 1. Загружаем наш обученный BPE-токенизатор
tokenizer = BPETokenizer()
tokenizer.load("models/bpe_tokenizer.json")
vocab_size = tokenizer.tokenizer.get_vocab_size()


class TokenGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        out = self.embedding(x)
        out, hidden = self.gru(out, hidden)
        out = self.fc(out)
        return out, hidden


def generate_text(model, tokenizer, start_str, gen_length=50, temperature=0.7):
    model.eval()

    # Кодируем стартовый промпт с помощью BPE токенизатора
    input_ids = tokenizer.encode(start_str)

    if not input_ids:
        return "Ошибка: промпт не удалось закодировать!"

    input_eval = torch.tensor([input_ids], dtype=torch.long)
    generated_ids = list(input_ids)
    hidden = None

    with torch.no_grad():
        # Прогоняем начальный контекст через модель, чтобы получить состояние скрытого слоя
        for i in range(len(input_ids) - 1):
            _, hidden = model(input_eval[:, i:i + 1], hidden)

        inp = input_eval[:, -1:]

        # Генерируем новые токены по одному
        for _ in range(gen_length):
            output, hidden = model(inp, hidden)
            logits = output[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            predicted_id = torch.multinomial(probs, num_samples=1).item()

            generated_ids.append(predicted_id)
            inp = torch.tensor([[predicted_id]], dtype=torch.long)

    # Декодируем все собранные токены обратно в читаемый текст
    return tokenizer.decode(generated_ids)


if __name__ == "__main__":
    model = TokenGRU(vocab_size, EMBED_DIM, HIDDEN_DIM)

    try:
        model.load_state_dict(torch.load("models/model.pt"))
        print("Веса модели успешно загружены!")
    except FileNotFoundError:
        print(
            "Внимание: файл весов 'models/model.pt' не найден. Модель будет генерировать случайный текст до обучения.")

    prompt = "Егор"
    print(f"\n--- Результат генерации по промпту '{prompt}': ---")
    print(generate_text(model, tokenizer, start_str=prompt, gen_length=30, temperature=0.7))