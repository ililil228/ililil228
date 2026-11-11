import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch_directml

from jsonl_streamer import stream_wiktextract_text
from streaming_dataset import StreamingWiktextractDataset
from tokenizer import BPETokenizer
from model import BPEModel


# ============================================================
# НАСТРОЙКИ
# ============================================================

JSONL_PATH = "resources/raw-wiktextract-data.jsonl"
TOKENIZER_PATH = "models/bpe_tokenizer.json"
MODEL_SAVE_PATH = "models/bpe_model.pt"

# Сколько текстовых записей брать из Wiktextract
MAX_LINES = 500_000

# Размер батча.
# Если RX 580 будет упираться в память — уменьшим до 16 или 8.
BATCH_SIZE = 16

# Learning rate
LEARNING_RATE = 0.0005

# Контекст модели
BLOCK_SIZE = 64

# Архитектура
EMBED_SIZE = 256
NUM_HEADS = 4
NUM_LAYERS = 4

# Сколько шагов обучения сделать
MAX_STEPS = 10_000

# Сохранять checkpoint каждые N шагов
SAVE_EVERY = 500


# ============================================================
# DEVICE
# ============================================================

device = torch_directml.device()


# ============================================================
# СОХРАНЕНИЕ CHECKPOINT
# ============================================================

def save_checkpoint(model, optimizer, step, loss, path):
    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True
    )

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        "step": step,
        "loss": loss,

        # Сохраняем архитектуру прямо в checkpoint
        "vocab_size": model.vocab_size,
        "embed_size": model.embed_size,
        "num_heads": model.num_heads,
        "num_layers": model.num_layers,
        "max_seq_len": model.max_seq_len,
    }

    torch.save(checkpoint, path)

    print()
    print("=" * 60)
    print("CHECKPOINT SAVED")
    print("=" * 60)
    print("File:", path)
    print("Step:", step)
    print("Loss:", loss)
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MY_AI — TRAIN TRANSFORMER")
    print("=" * 60)
    print()

    print("Device:", device)
    print("PyTorch:", torch.__version__)
    print()

    print("Architecture:")
    print("  vocab_size:", 16384)
    print("  embed_size:", EMBED_SIZE)
    print("  heads:", NUM_HEADS)
    print("  layers:", NUM_LAYERS)
    print("  context:", BLOCK_SIZE)
    print()

    print("Training:")
    print("  max lines:", MAX_LINES)
    print("  batch size:", BATCH_SIZE)
    print("  learning rate:", LEARNING_RATE)
    print("  max steps:", MAX_STEPS)
    print()

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    print("=" * 60)
    print("LOADING TOKENIZER")
    print("=" * 60)

    tokenizer = BPETokenizer(
        vocab_size=16384
    )

    tokenizer.load(
        TOKENIZER_PATH
    )

    vocab_size = tokenizer.get_vocab_size()

    print("Tokenizer:", TOKENIZER_PATH)
    print("Vocab:", vocab_size)
    print()

    if vocab_size != 16384:
        raise RuntimeError(
            f"Ожидался vocab 16384, "
            f"но tokenizer имеет {vocab_size}"
        )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print("=" * 60)
    print("CREATING DATASET")
    print("=" * 60)

    streamer = stream_wiktextract_text(
        JSONL_PATH,
        max_lines=MAX_LINES
    )

    dataset = StreamingWiktextractDataset(
        streamer,
        tokenizer,
        block_size=BLOCK_SIZE
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    print("Dataset:", JSONL_PATH)
    print("Block size:", BLOCK_SIZE)
    print("Batch size:", BATCH_SIZE)
    print()

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("=" * 60)
    print("CREATING MODEL")
    print("=" * 60)

    model = BPEModel(
        vocab_size=vocab_size,
        embed_size=EMBED_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=BLOCK_SIZE
    )

    model = model.to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameters:", f"{parameters:,}")
    print("Device:", device)
    print()

    # --------------------------------------------------------
    # LOSS / OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    start_step = 0

    if os.path.exists(MODEL_SAVE_PATH):

        print("=" * 60)
        print("CHECKPOINT FOUND")
        print("=" * 60)
        print(MODEL_SAVE_PATH)
        print()

        try:

            checkpoint = torch.load(
                MODEL_SAVE_PATH,
                map_location="cpu"
            )

            checkpoint_vocab = checkpoint.get(
                "vocab_size",
                None
            )

            print(
                "Checkpoint vocab:",
                checkpoint_vocab
            )

            # ------------------------------------------------
            # Проверяем совместимость
            # ------------------------------------------------

            if checkpoint_vocab != vocab_size:

                print()
                print(
                    "WARNING: checkpoint имеет другой vocab!"
                )

                print(
                    "Checkpoint:",
                    checkpoint_vocab
                )

                print(
                    "Current:",
                    vocab_size
                )

                print()
                print(
                    "Старый checkpoint НЕ загружаем."
                )

            else:

                model.load_state_dict(
                    checkpoint["state_dict"]
                )

                if "optimizer_state_dict" in checkpoint:

                    optimizer.load_state_dict(
                        checkpoint["optimizer_state_dict"]
                    )

                start_step = checkpoint.get(
                    "step",
                    0
                )

                print(
                    "Checkpoint loaded."
                )

                print(
                    "Starting from step:",
                    start_step
                )

        except Exception as e:

            print()
            print(
                "Не удалось загрузить checkpoint:"
            )

            print(e)

            print()
            print(
                "Начинаем обучение с нуля."
            )

    else:

        print(
            "Checkpoint не найден."
        )

        print(
            "Обучение с нуля."
        )

    print()

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    print("=" * 60)
    print("START TRAINING")
    print("=" * 60)
    print()

    step = start_step

    total_loss = 0.0
    start_time = time.time()

    while step < MAX_STEPS:

        for x, y in dataloader:

            if step >= MAX_STEPS:
                break

            # ------------------------------------------------
            # Переносим данные на DirectML
            # ------------------------------------------------

            x = x.to(device)
            y = y.to(device)

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            optimizer.zero_grad()

            logits, _ = model(x)

            # logits:
            # B, T, V
            #
            # CrossEntropy ожидает:
            # B*T, V
            #
            # targets:
            # B*T

            loss = criterion(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Optimizer
            # ------------------------------------------------

            optimizer.step()

            step += 1

            loss_value = loss.item()

            total_loss += loss_value

            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if step == 1 or step % 10 == 0:

                elapsed = time.time() - start_time

                steps_per_sec = (
                    step - start_step
                ) / max(elapsed, 0.001)

                print(
                    f"Step {step:6d} / {MAX_STEPS} | "
                    f"Loss: {loss_value:.4f} | "
                    f"Speed: {steps_per_sec:.2f} step/s"
                )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            if step % SAVE_EVERY == 0:

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    loss=loss_value,
                    path=MODEL_SAVE_PATH
                )

    # --------------------------------------------------------
    # FINAL SAVE
    # --------------------------------------------------------

    final_loss = loss.item()

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=step,
        loss=final_loss,
        path=MODEL_SAVE_PATH
    )

    print("=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)
    print()

    print("Final step:", step)
    print("Final loss:", final_loss)
    print("Model:", MODEL_SAVE_PATH)
    print()

    print("MY_AI Transformer готов.")


if __name__ == "__main__":
    main()