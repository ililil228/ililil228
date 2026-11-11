import os
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_directml

from tokenizer import BPETokenizer
from model import BPEModel
from streaming_dataset import StreamingWiktextractDataset
from jsonl_streamer import stream_wiktextract_text

# ============================================================
# CONFIG (УЛЬТИМАТИВНАЯ ВЕРСИЯ 44M - СВЕРХСТАБИЛЬНЫЙ DIRECTML)
# ============================================================
TOKENIZER_PATH = "models/bpe_tokenizer_v2.json"
CHECKPOINT_PATH = "models/bpe_model_v2.pt"

BLOCK_SIZE = 64

# Батчи подстроены так, чтобы загрузить 8 ГБ видеопамяти RX 580 на 100%
EFFECTIVE_BATCH_SIZE = 16
MICRO_BATCH_SIZE = 4

LEARNING_RATE = 3e-5
MAX_STEPS = 100000
SAVE_EVERY = 1000

WIKTEXTRACT_PATH = "resources/raw-wiktextract-data.jsonl"
MAX_WIKTEXTRACT_SAMPLES = 500000

# ============================================================
# DEVICE INITIALIZATION
# ============================================================
device = torch_directml.device()

print("=" * 60)
print("MY_AI V2 — ULTIMATE AMD GPU TRAINING (44M PARAMETERS)")
print("=" * 60)
print("Target Device:", device)
print()

# ============================================================
# TOKENIZER
# ============================================================
print("Loading V2 tokenizer...")
tokenizer = BPETokenizer()
tokenizer.load(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()

print("Vocab size:", vocab_size)
if vocab_size != 16384:
    raise RuntimeError(f"Wrong vocabulary size: expected 16384, got {vocab_size}")
print("Tokenizer: OK")
print()

# ============================================================
# MODEL DEFINITION
# ============================================================
print("Creating NEW ultimate 44M model...")
model = BPEModel(
    vocab_size=16384,
    embed_size=512,  # Жирный эмбеддинг для глубокого понимания контекста
    num_heads=8,  # 8 голов внимания
    num_layers=12,  # 12 слоев трансформера
    max_seq_len=64,
)

parameter_count = sum(p.numel() for p in model.parameters())
print("Parameters:", f"{parameter_count:,}")

# Переносим модель на видеокарту AMD
model = model.to(device)
print("Model moved to:", device)
print()

# ============================================================
# DATA LOADING
# ============================================================
print("Preparing dataset...")
streamer = stream_wiktextract_text(WIKTEXTRACT_PATH, max_lines=MAX_WIKTEXTRACT_SAMPLES)
dataset = StreamingWiktextractDataset(streamer=streamer, tokenizer=tokenizer, block_size=BLOCK_SIZE)
loader = DataLoader(dataset, batch_size=EFFECTIVE_BATCH_SIZE, num_workers=0)
print("Dataset: OK")
print()

# ============================================================
# OPTIMIZER
# ============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
print("Optimizer: AdamW")
print("Effective batch:", EFFECTIVE_BATCH_SIZE)
print("Micro batch:", MICRO_BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print()

# ============================================================
# TRAINING LOOP
# ============================================================
model.train()
step = 0

print("=" * 60)
print("START TRAINING FROM STEP 0 (PURE GPU RUN)")
print("=" * 60)
print()

start_time = time.time()

try:
    for x, y in loader:
        if step >= MAX_STEPS:
            break

        optimizer.zero_grad(set_to_none=True)
        batch_size = x.size(0)
        num_micro = math.ceil(batch_size / MICRO_BATCH_SIZE)
        total_loss = 0.0

        for start in range(0, batch_size, MICRO_BATCH_SIZE):
            end = min(start + MICRO_BATCH_SIZE, batch_size)

            # Переносим входные данные на видеокарту AMD
            micro_x = x[start:end].to(device)
            micro_y = y[start:end].to(device)

            logits, _ = model(micro_x)

            # ИСПРАВЛЕНО ДЛЯ DIRECTML: Используем .view() вместо .reshape().
            # Это исключает скрытое копирование в RAM и не дает вылетать вычислениям на CPU.
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                micro_y.view(-1),
                ignore_index=0
            )

            scaled_loss = loss / num_micro
            scaled_loss.backward()

            total_loss += loss.item()

        # Жесткая обрезка градиентов для защиты стабильности весов
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        step += 1

        average_loss = total_loss / num_micro

        # Вывод статистики в консоль каждые 10 шагов
        if step == 1 or step % 10 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / max(elapsed, 0.001)
            steps_per_min = steps_per_sec * 60

            print(
                f"Step {step:6d} / {MAX_STEPS} | "
                f"Loss: {average_loss:.4f} | "
                f"Speed: {steps_per_min:.2f} step/min"
            )

        # ============================================================
        # SAVE CHECKPOINT
        # ============================================================
        if step % SAVE_EVERY == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "loss": average_loss,
                "vocab_size": vocab_size,
                "embed_size": 512,
                "num_heads": 8,
                "num_layers": 12,
                "max_seq_len": 64,
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            print(f"\n[💾 Checkpoint saved: {CHECKPOINT_PATH} at step {step}]\n")

except KeyboardInterrupt:
    print("\nTraining interrupted.")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "vocab_size": vocab_size,
        "embed_size": 512,
        "num_heads": 8,
        "num_layers": 12,
        "max_seq_len": 64,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"Checkpoint saved on interrupt: {CHECKPOINT_PATH}")

print()
print("=" * 60)
print("TRAINING FINISHED")
print("=" * 60)
print("Steps:", step)
print("Checkpoint:", CHECKPOINT_PATH)
