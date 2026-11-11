import torch
import torch_directml

from tokenizer import BPETokenizer
from model import BPEModel


TOKENIZER_PATH = "models/bpe_tokenizer_v2.json"
CHECKPOINT_PATH = "models/bpe_model_v2.pt"

MAX_NEW_TOKENS = 100


# =========================
# DEVICE
# =========================

device = torch_directml.device()

print("=" * 60)
print("MY_AI V2 GENERATION TEST")
print("=" * 60)
print("Device:", device)
print()


# =========================
# TOKENIZER
# =========================

tokenizer = BPETokenizer()
tokenizer.load(TOKENIZER_PATH)

vocab_size = tokenizer.get_vocab_size()

print("Tokenizer vocab:", vocab_size)

if vocab_size != 16384:
    raise RuntimeError(
        f"Wrong vocab size: {vocab_size}"
    )


# =========================
# MODEL
# =========================

model = BPEModel(
    vocab_size=16384,
    embed_size=256,
    num_heads=4,
    num_layers=4,
    max_seq_len=64,
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Checkpoint step:", checkpoint.get("step", "?"))
print("Model: OK")
print()


# =========================
# GENERATION
# =========================

def generate(prompt, max_new_tokens=100):

    ids = tokenizer.encode(prompt)

    if not ids:
        return ""

    # Ограничиваем контекст размером модели
    ids = ids[-64:]

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Модель принимает максимум 64 токена
            x_cond = x[:, -64:]

            logits, _ = model(x_cond)

            # Последний токен
            logits = logits[:, -1, :]

            # Пока используем простейший greedy decoding
            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

            x = torch.cat(
                [x, next_token],
                dim=1
            )

    generated_ids = x[0].detach().cpu().tolist()

    return tokenizer.decode(generated_ids)


# =========================
# TEST
# =========================

print("=" * 60)
print("GENERATION")
print("=" * 60)

prompt = "Привет! Как дела?"

print("Prompt:")
print(prompt)
print()
print("Generating...")
print()

try:
    result = generate(
        prompt,
        MAX_NEW_TOKENS
    )

    print("RESULT:")
    print(result)

except Exception as e:

    print()
    print("GENERATION ERROR:")
    print(type(e).__name__)
    print(e)

print()
print("=" * 60)
print("TEST FINISHED")
print("=" * 60)