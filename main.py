import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1. Конфигурация
SEQ_LEN = 32
BATCH_SIZE = 16
EMBED_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 50
LR = 0.003

# 2. Подготовка данных
with open("data/story.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
data_indices = [char_to_idx[ch] for ch in text]

class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.tensor(data, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

dataset = TextDataset(data_indices, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. Модель GRU
class CharacterGRU(nn.Module):
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

model = CharacterGRU(vocab_size, EMBED_DIM, HIDDEN_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# 4. Обучение
print(f"Начало обучения (Размер словаря: {vocab_size} символов)...")
model.train()

for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        output, _ = model(x_batch)
        loss = criterion(output.reshape(-1, vocab_size), y_batch.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0 or epoch == 1:
        avg_loss = total_loss / len(dataloader)
        print(f"Эпоха {epoch}/{EPOCHS} | Loss: {avg_loss:.4f}")

# 5. Сохранение весов
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model.pt")
print("Обучение завершено! Модель сохранена в models/model.pt")