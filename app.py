import os
import time
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch_directml
import streamlit as st
import psutil
import pandas as pd
from torch.utils.data import DataLoader

from tokenizer import BPETokenizer
from jsonl_streamer import (
    StreamingWiktextractDataset,
    stream_dialogue_jsonl,
)
from model import BPEModel


# ============================================================
# MY_AI V2 - Streamlit interface
# ============================================================

st.set_page_config(
    page_title="MY_AI V2",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RESOURCES_DIR = BASE_DIR / "resources"
MODELS_DIR = BASE_DIR / "models"

WIKTEXTRACT_PATH = (
    RESOURCES_DIR / "raw-wiktextract-data.jsonl"
)

CULTURA_PATH = (
    RESOURCES_DIR / "cultura_ru_subset.txt"
)

DIALOGUE_PATH = (
    RESOURCES_DIR / "dataset.jsonl"
)

TOKENIZER_PATH = (
    MODELS_DIR / "bpe_tokenizer_v2.json"
)

MODEL_SAVE_PATH = (
    MODELS_DIR / "bpe_model_v2.pt"
)

# Legacy files are intentionally NOT used.
LEGACY_TOKENIZER_PATH = (
    MODELS_DIR / "bpe_tokenizer.json"
)

LEGACY_MODEL_PATH = (
    MODELS_DIR / "bpe_model.pt"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

VOCAB_SIZE = 16384

EMBED_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 4
FFN_DIM = 1024
MAX_SEQ_LEN = 64

MICRO_BATCH_SIZE = 4

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01

DEFAULT_MAX_STEPS = 10000
CHECKPOINT_EVERY = 100


# ============================================================
# DEVICE
# ============================================================

try:
    DEVICE = torch_directml.device()
    DEVICE_NAME = "AMD GPU / DirectML"
except Exception:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"


# ============================================================
# HELPERS
# ============================================================

def format_size(path: Path) -> str:
    if not path.exists():
        return "нет файла"

    size = path.stat().st_size

    if size < 1024:
        return f"{size} B"

    if size < 1024 ** 2:
        return f"{size / 1024:.1f} KB"

    if size < 1024 ** 3:
        return f"{size / 1024 ** 2:.1f} MB"

    return f"{size / 1024 ** 3:.2f} GB"


def move_to_cpu(obj):
    """
    Recursively move tensors to CPU before saving.
    """

    if torch.is_tensor(obj):
        return obj.detach().cpu()

    if isinstance(obj, dict):
        return {
            key: move_to_cpu(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            move_to_cpu(value)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            move_to_cpu(value)
            for value in obj
        )

    return obj


def load_checkpoint(path: Path):

    if not path.exists():
        return None

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def get_state_dict(checkpoint):

    if checkpoint is None:
        return None

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if state_dict is None:
        state_dict = checkpoint.get(
            "state_dict"
        )

    if state_dict is None:

        if isinstance(checkpoint, dict):

            possible_state = {
                key: value
                for key, value in checkpoint.items()
                if torch.is_tensor(value)
            }

            if possible_state:
                return possible_state

    return state_dict


def get_checkpoint_step(checkpoint):

    if not checkpoint:
        return 0

    for key in (
        "step",
        "steps",
        "global_step",
    ):

        value = checkpoint.get(key)

        if value is not None:

            try:
                return int(value)

            except Exception:
                pass

    return 0


def get_checkpoint_loss(checkpoint):

    if not checkpoint:
        return None

    for key in (
        "loss",
        "last_loss",
        "train_loss",
    ):

        value = checkpoint.get(key)

        if value is not None:

            try:
                return float(value)

            except Exception:
                pass

    return None


def build_model():

    model = BPEModel(
        vocab_size=VOCAB_SIZE,
        embed_size=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
    )

    return model


@st.cache_resource
def get_tokenizer():

    if not TOKENIZER_PATH.exists():

        raise FileNotFoundError(
            "Не найден V2 токенизатор:\n"
            f"{TOKENIZER_PATH}"
        )

    tokenizer = BPETokenizer(
        vocab_size=VOCAB_SIZE
    )

    tokenizer.load(
        str(TOKENIZER_PATH)
    )

    return tokenizer


def load_model_for_inference():

    if not MODEL_SAVE_PATH.exists():
        return None, None

    checkpoint = load_checkpoint(
        MODEL_SAVE_PATH
    )

    model = build_model()

    state_dict = get_state_dict(
        checkpoint
    )

    if state_dict is None:

        raise RuntimeError(
            "В checkpoint нет "
            "model_state_dict/state_dict."
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    st.write(
        "🔧 DEVICE:",
        DEVICE
    )

    try:

        first_param = next(
            model.parameters()
        )

        st.write(
            "🧠 MODEL DEVICE:",
            first_param.device
        )

    except Exception as e:

        st.write(
            "Ошибка проверки device:",
            e
        )

    return model, checkpoint


def save_checkpoint(
    model,
    optimizer,
    step,
    loss,
):

    checkpoint = {

        "step":
            int(step),

        "loss":
            float(loss),

        "model_state_dict":
            move_to_cpu(
                model.state_dict()
            ),

        "optimizer_state_dict":
            move_to_cpu(
                optimizer.state_dict()
            ),

        "vocab_size":
            VOCAB_SIZE,

        "embed_dim":
            EMBED_DIM,

        "num_heads":
            NUM_HEADS,

        "num_layers":
            NUM_LAYERS,

        "ffn_dim":
            FFN_DIM,

        "max_seq_len":
            MAX_SEQ_LEN,
    }

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        MODEL_SAVE_PATH,
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.replace(
        "\x00",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# GENERIC TEXT FILE STREAM
# ============================================================

def stream_text_file(path: Path):

    if not path.exists():
        return

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:

            text = clean_text(
                line
            )

            if text:
                yield text


# ============================================================
# WIKTEXTRACT STREAM
# ============================================================

def stream_wiktextract(max_items):

    if not WIKTEXTRACT_PATH.exists():
        return

    dataset = StreamingWiktextractDataset(
        str(WIKTEXTRACT_PATH),
        max_lines=max_items,
    )

    for item in dataset:

        if isinstance(item, str):

            text = clean_text(
                item
            )

        elif isinstance(item, dict):

            text = clean_text(
                item.get("text")
                or item.get("gloss")
                or item.get("definition")
                or ""
            )

        else:

            text = clean_text(
                str(item)
            )

        if text:
            yield text


# ============================================================
# LOCAL CULTURA STREAM
# ============================================================

def stream_cultura_local(max_items):

    if not CULTURA_PATH.exists():
        return

    count = 0

    for text in stream_text_file(
        CULTURA_PATH
    ):

        yield text

        count += 1

        if (
            max_items
            and count >= max_items
        ):
            break


# ============================================================
# LOCAL DIALOGUE STREAM
# ============================================================

def stream_dialogue(max_items):

    if not DIALOGUE_PATH.exists():
        return

    yield from stream_dialogue_jsonl(
        str(DIALOGUE_PATH),
        max_lines=max_items,
    )


# ============================================================
# COMBINED STREAM
# ============================================================

def combined_stream(
    source,
    wiktextract_items,
    cultura_items,
    dialogue_items,
):

    # --------------------------------------------------------
    # Wiktextract
    # --------------------------------------------------------

    if source == "Wiktextract":

        yield from stream_wiktextract(
            wiktextract_items
        )

        return

    # --------------------------------------------------------
    # Cultura
    # --------------------------------------------------------

    if source == "Cultura":

        yield from stream_cultura_local(
            cultura_items
        )

        return

    # --------------------------------------------------------
    # Dialogues
    # --------------------------------------------------------

    if source == "Диалоги":

        yield from stream_dialogue(
            dialogue_items
        )

        return

    # --------------------------------------------------------
    # Cultura + Wiktextract
    # --------------------------------------------------------

    if source == "Cultura + Wiktextract":

        yield from stream_cultura_local(
            cultura_items
        )

        yield from stream_wiktextract(
            wiktextract_items
        )

        return

    # --------------------------------------------------------
    # Wiktextract + Dialogues
    # --------------------------------------------------------

    if source == "Wiktextract + Диалоги":

        yield from stream_wiktextract(
            wiktextract_items
        )

        yield from stream_dialogue(
            dialogue_items
        )

        return

    # --------------------------------------------------------
    # Everything
    # --------------------------------------------------------

    if source == "Cultura + Wiktextract + Диалоги":

        yield from stream_cultura_local(
            cultura_items
        )

        yield from stream_wiktextract(
            wiktextract_items
        )

        yield from stream_dialogue(
            dialogue_items
        )

        return


# ============================================================
# BATCH DATASET
# ============================================================

class TokenBatchDataset(
    torch.utils.data.IterableDataset
):

    def __init__(
        self,
        tokenizer,
        text_stream,
    ):

        super().__init__()

        self.tokenizer = tokenizer
        self.text_stream = text_stream

    def __iter__(self):

        for text in self.text_stream:

            try:

                ids = self.tokenizer.encode(
                    text
                )

                if len(ids) < 2:
                    continue

                # Split text into fixed-size chunks.
                for start in range(
                    0,
                    len(ids) - 1,
                    MAX_SEQ_LEN,
                ):

                    chunk = ids[
                        start:
                        start + MAX_SEQ_LEN + 1
                    ]

                    if len(chunk) < 2:
                        continue

                    x = chunk[:-1]
                    y = chunk[1:]

                    if len(x) < MAX_SEQ_LEN:

                        pad_id = 0

                        x = x + [
                            pad_id
                        ] * (
                            MAX_SEQ_LEN
                            - len(x)
                        )

                        y = y + [
                            pad_id
                        ] * (
                            MAX_SEQ_LEN
                            - len(y)
                        )

                    yield (
                        torch.tensor(
                            x,
                            dtype=torch.long,
                        ),
                        torch.tensor(
                            y,
                            dtype=torch.long,
                        ),
                    )

            except Exception:
                continue


# ============================================================
# DATALOADER
# ============================================================

def make_dataloader(
    tokenizer,
    source,
    wiktextract_items,
    cultura_items,
    dialogue_items,
    batch_size,
):

    text_stream = combined_stream(
        source,
        wiktextract_items,
        cultura_items,
        dialogue_items,
    )

    dataset = TokenBatchDataset(
        tokenizer,
        text_stream,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
    )


# ============================================================
# GENERATION
# ============================================================

@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    temperature,
    top_k,
):

    ids = tokenizer.encode(
        prompt
    )

    if not ids:
        return prompt

    generated = list(ids)

    for _ in range(
        max_new_tokens
    ):

        context = generated[
            -MAX_SEQ_LEN:
        ]

        input_ids = torch.tensor(
            [context],
            dtype=torch.long,
            device=DEVICE,
        )

        logits, _ = model(
            input_ids
        )

        next_logits = logits[
            0,
            -1,
            :,
        ]

        temperature = max(
            float(temperature),
            0.05,
        )

        next_logits = (
            next_logits
            / temperature
        )

        if top_k > 0:

            values, indices = torch.topk(
                next_logits,
                min(
                    top_k,
                    next_logits.numel(),
                ),
            )

            filtered = torch.full_like(
                next_logits,
                float("-inf"),
            )

            filtered[
                indices
            ] = values

            next_logits = filtered

        probabilities = torch.softmax(
            next_logits,
            dim=-1,
        )

        if not torch.isfinite(
            probabilities
        ).all():

            next_token = torch.argmax(
                next_logits
            ).item()

        else:

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            ).item()

        generated.append(
            int(next_token)
        )

        if next_token == 3:
            break

    try:

        return tokenizer.decode(
            generated
        )

    except Exception:

        return prompt


# ============================================================
# SESSION STATE
# ============================================================

if "training_running" not in st.session_state:

    st.session_state.training_running = False


if "last_training_loss" not in st.session_state:

    st.session_state.last_training_loss = None


# ============================================================
# UI
# ============================================================

st.title(
    "🤖 MY_AI V2"
)

st.caption(
    "Custom BPE tokenizer + decoder-only Transformer | "
    "AMD RX 580 / DirectML"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Система"
    )

    st.write(
        f"**Устройство:** {DEVICE_NAME}"
    )

    st.write(
        f"**Python PID:** {os.getpid()}"
    )

    ram = psutil.virtual_memory()

    st.write(
        f"**RAM:** "
        f"{ram.used / 1024**3:.1f} / "
        f"{ram.total / 1024**3:.1f} GB"
    )

    st.divider()

    st.header(
        "MY_AI V2"
    )

    st.write(
        f"Vocab: `{VOCAB_SIZE}`"
    )

    st.write(
        f"Embedding: `{EMBED_DIM}`"
    )

    st.write(
        f"Layers: `{NUM_LAYERS}`"
    )

    st.write(
        f"Heads: `{NUM_HEADS}`"
    )

    st.write(
        f"Context: `{MAX_SEQ_LEN}`"
    )

    st.write(
        f"Micro-batch: `{MICRO_BATCH_SIZE}`"
    )

    st.divider()

    if TOKENIZER_PATH.exists():

        st.success(
            "V2 tokenizer: "
            f"{format_size(TOKENIZER_PATH)}"
        )

    else:

        st.error(
            "V2 tokenizer не найден"
        )

    if MODEL_SAVE_PATH.exists():

        st.success(
            "V2 checkpoint: "
            f"{format_size(MODEL_SAVE_PATH)}"
        )

    else:

        st.warning(
            "V2 checkpoint пока отсутствует"
        )

    st.divider()

    st.subheader(
        "Данные"
    )

    if WIKTEXTRACT_PATH.exists():

        st.success(
            "Wiktextract: "
            f"{format_size(WIKTEXTRACT_PATH)}"
        )

    else:

        st.error(
            "Wiktextract не найден"
        )

    if CULTURA_PATH.exists():

        st.success(
            "Cultura local: "
            f"{format_size(CULTURA_PATH)}"
        )

    else:

        st.warning(
            "Cultura local не найден"
        )

    if DIALOGUE_PATH.exists():

        st.success(
            "Диалоги: "
            f"{format_size(DIALOGUE_PATH)}"
        )

    else:

        st.warning(
            "dataset.jsonl не найден"
        )

    st.caption(
        "Все источники данных читаются "
        "локально и потоково."
    )

    st.caption(
        "Старые bpe_tokenizer.json / "
        "bpe_model.pt этим приложением "
        "не используются."
    )


# ============================================================
# TABS
# ============================================================

tab_train, tab_chat, tab_model = st.tabs(
    [
        "🔥 Обучение",
        "💬 Чат / генерация",
        "🧠 Модель",
    ]
)


# ============================================================
# TRAINING
# ============================================================

with tab_train:

    st.header(
        "Обучение MY_AI V2"
    )

    if not TOKENIZER_PATH.exists():

        st.error(
            "Не найден "
            "`models/bpe_tokenizer_v2.json`."
        )

        st.stop()

    tokenizer = get_tokenizer()

    st.success(
        "V2 токенизатор загружен. "
        f"Размер словаря: {VOCAB_SIZE}"
    )

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        source = st.selectbox(
            "Источник данных:",
            [
                "Cultura",
                "Wiktextract",
                "Диалоги",
                "Cultura + Wiktextract",
                "Wiktextract + Диалоги",
                "Cultura + Wiktextract + Диалоги",
            ],
            index=2,
        )

    with col2:

        wiktextract_items = st.number_input(
            "Wiktextract элементов:",
            min_value=100,
            max_value=5_000_000,
            value=500_000,
            step=100_000,
        )

    with col3:

        cultura_items = st.number_input(
            "Cultura элементов:",
            min_value=100,
            max_value=1_000_000,
            value=100_000,
            step=10_000,
        )

    dialogue_items = st.number_input(
        "Диалогов из dataset.jsonl:",
        min_value=100,
        max_value=5_000_000,
        value=100_000,
        step=10_000,
    )

    # --------------------------------------------------------
    # Data availability
    # --------------------------------------------------------

    if source in (
        "Cultura",
        "Cultura + Wiktextract",
        "Cultura + Wiktextract + Диалоги",
    ):

        if not CULTURA_PATH.exists():

            st.warning(
                "⚠️ Выбран источник Cultura, "
                "но локальный файл не найден:\n"
                f"`{CULTURA_PATH}`\n\n"
                "Интернет-версия Cultura отключена."
            )

    if source in (
        "Диалоги",
        "Wiktextract + Диалоги",
        "Cultura + Wiktextract + Диалоги",
    ):

        if not DIALOGUE_PATH.exists():

            st.error(
                "❌ Выбраны диалоги, "
                "но `resources/dataset.jsonl` "
                "не найден."
            )

    if source in (
        "Wiktextract",
        "Cultura + Wiktextract",
        "Wiktextract + Диалоги",
        "Cultura + Wiktextract + Диалоги",
    ):

        if not WIKTEXTRACT_PATH.exists():

            st.error(
                "❌ Выбран Wiktextract, "
                "но файл не найден:\n"
                f"`{WIKTEXTRACT_PATH}`"
            )

    st.divider()

    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    batch_size = st.selectbox(
        "Размер батча (Batch Size):",
        [
            16,
            32,
            64,
            128,
            256,
            220,
        ],
        index=0,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        learning_rate = st.number_input(
            "Learning rate:",
            min_value=0.000001,
            max_value=0.01,
            value=LEARNING_RATE,
            format="%.6f",
        )

    with col2:

        max_steps = st.number_input(
            "Максимум шагов:",
            min_value=1,
            max_value=1_000_000,
            value=DEFAULT_MAX_STEPS,
            step=100,
        )

    with col3:

        save_every = st.number_input(
            "Сохранять каждые N шагов:",
            min_value=1,
            max_value=10_000,
            value=CHECKPOINT_EVERY,
            step=10,
        )

    # --------------------------------------------------------
    # Existing checkpoint
    # --------------------------------------------------------

    checkpoint = None
    start_step = 0

    if MODEL_SAVE_PATH.exists():

        try:

            checkpoint = load_checkpoint(
                MODEL_SAVE_PATH
            )

            start_step = (
                get_checkpoint_step(
                    checkpoint
                )
            )

            last_loss = (
                get_checkpoint_loss(
                    checkpoint
                )
            )

            st.info(
                f"Найден V2 checkpoint: "
                f"шаг **{start_step}**"
                +
                (
                    f", loss **{last_loss:.4f}**"
                    if last_loss is not None
                    else ""
                )
            )

        except Exception as e:

            st.error(
                "Не удалось прочитать "
                f"checkpoint: {e}"
            )

    st.write(
        f"Micro-batch = **{MICRO_BATCH_SIZE}**. "
        "Большой batch автоматически разбивается "
        "на небольшие микро-батчи."
    )

    start_training = st.button(
        "🚀 Начать / продолжить обучение",
        type="primary",
        width="stretch",
    )

    # ========================================================
    # START TRAINING
    # ========================================================

    if start_training:

        st.session_state.training_running = True

        try:

            # ------------------------------------------------
            # Validate selected sources
            # ------------------------------------------------

            if source == "Cultura":

                if not CULTURA_PATH.exists():

                    raise RuntimeError(
                        "Выбрана Cultura, "
                        "но локальный файл "
                        "cultura_ru_subset.txt "
                        "не найден."
                    )

            if source == "Диалоги":

                if not DIALOGUE_PATH.exists():

                    raise RuntimeError(
                        "Выбраны диалоги, "
                        "но dataset.jsonl "
                        "не найден."
                    )

            if source in (
                "Wiktextract",
                "Cultura + Wiktextract",
                "Wiktextract + Диалоги",
                "Cultura + Wiktextract + Диалоги",
            ):

                if not WIKTEXTRACT_PATH.exists():

                    raise RuntimeError(
                        "Выбран Wiktextract, "
                        "но raw-wiktextract-data.jsonl "
                        "не найден."
                    )

            if source in (
                "Cultura + Wiktextract",
                "Cultura + Wiktextract + Диалоги",
            ):

                if not CULTURA_PATH.exists():

                    raise RuntimeError(
                        "Выбрана Cultura, "
                        "но локальный файл "
                        "cultura_ru_subset.txt "
                        "не найден."
                    )

            if source in (
                "Wiktextract + Диалоги",
                "Cultura + Wiktextract + Диалоги",
            ):

                if not DIALOGUE_PATH.exists():

                    raise RuntimeError(
                        "Выбраны диалоги, "
                        "но dataset.jsonl "
                        "не найден."
                    )

            # ------------------------------------------------
            # Build model
            # ------------------------------------------------

            model = build_model()

            # ------------------------------------------------
            # Load checkpoint
            # ------------------------------------------------

            if checkpoint is not None:

                state_dict = get_state_dict(
                    checkpoint
                )

                if state_dict is None:

                    raise RuntimeError(
                        "Checkpoint найден, "
                        "но model_state_dict/"
                        "state_dict отсутствует."
                    )

                model.load_state_dict(
                    state_dict,
                    strict=True,
                )

            model.to(
                DEVICE
            )

            model.train()

            # ------------------------------------------------
            # Optimizer
            # ------------------------------------------------

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(
                    learning_rate
                ),
                weight_decay=WEIGHT_DECAY,
            )

            # ------------------------------------------------
            # Restore optimizer
            # ------------------------------------------------

            if checkpoint is not None:

                optimizer_state = (
                    checkpoint.get(
                        "optimizer_state_dict"
                    )
                )

                if optimizer_state is not None:

                    try:

                        optimizer.load_state_dict(
                            optimizer_state
                        )

                    except Exception as e:

                        st.warning(
                            "Состояние оптимизатора "
                            "не загрузилось. "
                            "Создан новый optimizer.\n"
                            f"{e}"
                        )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            criterion = nn.CrossEntropyLoss(
                ignore_index=0
            )

            # ------------------------------------------------
            # DataLoader
            # ------------------------------------------------

            dataloader = make_dataloader(
                tokenizer=tokenizer,
                source=source,
                wiktextract_items=int(
                    wiktextract_items
                ),
                cultura_items=int(
                    cultura_items
                ),
                dialogue_items=int(
                    dialogue_items
                ),
                batch_size=int(
                    batch_size
                ),
            )

            data_iterator = iter(
                dataloader
            )

            # ------------------------------------------------
            # Training settings
            # ------------------------------------------------

            total_steps = int(
                max_steps
            )

            step = int(
                start_step
            )

            running_loss = (
                get_checkpoint_loss(
                    checkpoint
                )
                if checkpoint is not None
                else 0.0
            )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            loss_history = []
            step_history = []

            training_start_time = time.time()

            # ------------------------------------------------
            # UI placeholders
            # ------------------------------------------------

            progress = st.progress(
                min(
                    1.0,
                    step / total_steps
                    if total_steps > 0
                    else 0,
                )
            )

            status_placeholder = st.empty()

            metric_col1, metric_col2, metric_col3, metric_col4 = (
                st.columns(4)
            )

            with metric_col1:
                loss_metric = st.empty()

            with metric_col2:
                step_metric = st.empty()

            with metric_col3:
                speed_metric = st.empty()

            with metric_col4:
                eta_metric = st.empty()

            st.subheader(
                "📉 Loss"
            )

            chart_placeholder = st.empty()

            # ------------------------------------------------
            # Initial UI
            # ------------------------------------------------

            loss_metric.metric(
                "Loss",
                (
                    f"{running_loss:.4f}"
                    if running_loss
                    else "—"
                ),
            )

            step_metric.metric(
                "Шаг",
                f"{step:,} / {total_steps:,}",
            )

            speed_metric.metric(
                "Скорость",
                "—",
            )

            eta_metric.metric(
                "Осталось",
                "—",
            )

            # =================================================
            # MAIN TRAINING LOOP
            # =================================================

            while step < total_steps:

                # ---------------------------------------------
                # Reset gradients
                # ---------------------------------------------

                optimizer.zero_grad(
                    set_to_none=True
                )

                micro_losses = []

                # ---------------------------------------------
                # Get ONE batch
                # ---------------------------------------------

                try:

                    x, y = next(
                        data_iterator
                    )

                except StopIteration:

                    data_iterator = iter(
                        dataloader
                    )

                    try:

                        x, y = next(
                            data_iterator
                        )

                    except StopIteration:

                        raise RuntimeError(
                            "Источник данных "
                            "не дал ни одного "
                            "обучающего примера."
                        )

                x = x.to(
                    DEVICE
                )

                y = y.to(
                    DEVICE
                )

                if step == start_step:

                    st.write(
                        "📦 INPUT DEVICE:",
                        x.device
                    )

                    st.write(
                        "🎯 TARGET DEVICE:",
                        y.device
                    )

                # ---------------------------------------------
                # Number of micro-batches
                # ---------------------------------------------

                current_batch = x.shape[0]

                num_micro = max(
                    1,
                    (
                        current_batch
                        + MICRO_BATCH_SIZE
                        - 1
                    )
                    // MICRO_BATCH_SIZE,
                )

                # ---------------------------------------------
                # Micro-batch loop
                # ---------------------------------------------

                for start in range(
                    0,
                    current_batch,
                    MICRO_BATCH_SIZE,
                ):

                    end = min(
                        start
                        + MICRO_BATCH_SIZE,
                        current_batch,
                    )

                    xb = x[
                        start:end
                    ]

                    yb = y[
                        start:end
                    ]

                    # -----------------------------------------
                    # Forward
                    # -----------------------------------------

                    logits, _ = model(
                        xb
                    )

                    # -----------------------------------------
                    # Loss
                    # -----------------------------------------

                    loss = criterion(
                        logits.reshape(
                            -1,
                            VOCAB_SIZE,
                        ),
                        yb.reshape(
                            -1
                        ),
                    )

                    # -----------------------------------------
                    # Normalize gradients
                    # -----------------------------------------

                    scaled_loss = (
                        loss / num_micro
                    )

                    scaled_loss.backward()

                    micro_losses.append(
                        float(
                            loss.detach()
                            .cpu()
                        )
                    )

                # ---------------------------------------------
                # Gradient clipping
                # ---------------------------------------------

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                # ---------------------------------------------
                # Optimizer step
                # ---------------------------------------------

                optimizer.step()

                # ---------------------------------------------
                # Increase global step
                # ---------------------------------------------

                step += 1

                # ---------------------------------------------
                # Current loss
                # ---------------------------------------------

                if micro_losses:

                    running_loss = (
                        sum(
                            micro_losses
                        )
                        / len(
                            micro_losses
                        )
                    )

                # ---------------------------------------------
                # Save statistics
                # ---------------------------------------------

                loss_history.append(
                    running_loss
                )

                step_history.append(
                    step
                )

                # ---------------------------------------------
                # Time
                # ---------------------------------------------

                elapsed = (
                    time.time()
                    - training_start_time
                )

                steps_done_this_run = (
                    step
                    - start_step
                )

                if elapsed > 0:

                    steps_per_second = (
                        steps_done_this_run
                        / elapsed
                    )

                else:

                    steps_per_second = 0.0

                steps_per_minute = (
                    steps_per_second
                    * 60
                )

                # ---------------------------------------------
                # ETA
                # ---------------------------------------------

                remaining_steps = max(
                    0,
                    total_steps - step,
                )

                if (
                    steps_per_second > 0
                ):

                    remaining_seconds = (
                        remaining_steps
                        / steps_per_second
                    )

                else:

                    remaining_seconds = 0

                remaining_hours = int(
                    remaining_seconds
                    // 3600
                )

                remaining_minutes = int(
                    (
                        remaining_seconds
                        % 3600
                    )
                    // 60
                )

                remaining_secs = int(
                    remaining_seconds
                    % 60
                )

                if remaining_hours > 0:

                    eta_text = (
                        f"{remaining_hours} ч "
                        f"{remaining_minutes:02d} мин"
                    )

                else:

                    eta_text = (
                        f"{remaining_minutes} мин "
                        f"{remaining_secs:02d} сек"
                    )

                # ---------------------------------------------
                # Progress
                # ---------------------------------------------

                progress_value = min(
                    1.0,
                    step / total_steps
                    if total_steps > 0
                    else 1.0,
                )

                progress.progress(
                    progress_value
                )

                # ---------------------------------------------
                # Status
                # ---------------------------------------------

                status_placeholder.write(
                    f"🔥 **Обучение идёт** — "
                    f"step `{step:,}` / "
                    f"`{total_steps:,}`"
                )

                # ---------------------------------------------
                # Metrics
                # ---------------------------------------------

                loss_metric.metric(
                    "Loss",
                    f"{running_loss:.4f}",
                )

                step_metric.metric(
                    "Шаг",
                    f"{step:,} / "
                    f"{total_steps:,}",
                )

                speed_metric.metric(
                    "Скорость",
                    f"{steps_per_minute:.2f} "
                    "шаг/мин",
                )

                eta_metric.metric(
                    "Осталось",
                    eta_text,
                )

                # ---------------------------------------------
                # Update graph every 10 steps
                # ---------------------------------------------

                if (
                    step == 1
                    or step % 10 == 0
                    or step == total_steps
                ):

                    loss_df = pd.DataFrame(
                        {
                            "Loss":
                                loss_history
                        },
                        index=step_history,
                    )

                    loss_df.index.name = (
                        "Step"
                    )

                    chart_placeholder.line_chart(
                        loss_df,
                        width="stretch",
                    )

                # ---------------------------------------------
                # Session state
                # ---------------------------------------------

                st.session_state.last_training_loss = (
                    running_loss
                )

                # ---------------------------------------------
                # Checkpoint
                # ---------------------------------------------

                if (
                    step
                    % int(save_every)
                    == 0
                    or step >= total_steps
                ):

                    save_checkpoint(
                        model,
                        optimizer,
                        step,
                        running_loss,
                    )

                    status_placeholder.write(
                        f"💾 **Checkpoint сохранён** — "
                        f"step `{step:,}` / "
                        f"`{total_steps:,}`"
                    )

            # =================================================
            # TRAINING FINISHED
            # =================================================

            save_checkpoint(
                model,
                optimizer,
                step,
                running_loss,
            )

            st.session_state.training_running = (
                False
            )

            total_elapsed = (
                time.time()
                - training_start_time
            )

            total_minutes = int(
                total_elapsed // 60
            )

            total_seconds = int(
                total_elapsed % 60
            )

            st.success(
                f"🎉 Обучение завершено!\n\n"
                f"Шаг: **{step:,} / "
                f"{total_steps:,}**\n\n"
                f"Loss: **{running_loss:.4f}**\n\n"
                f"Время этой сессии: "
                f"**{total_minutes} мин "
                f"{total_seconds:02d} сек**"
            )

        # =====================================================
        # INTERRUPT
        # =====================================================

        except KeyboardInterrupt:

            st.session_state.training_running = (
                False
            )

            st.warning(
                "⚠️ Обучение прервано."
            )

        # =====================================================
        # ERROR
        # =====================================================

        except Exception as e:

            st.session_state.training_running = (
                False
            )

            st.exception(e)


# ============================================================
# CHAT / GENERATION
# ============================================================

with tab_chat:

    st.header(
        "💬 Генерация текста"
    )

    if not TOKENIZER_PATH.exists():

        st.error(
            "Нет "
            "`models/bpe_tokenizer_v2.json`."
        )

    elif not MODEL_SAVE_PATH.exists():

        st.warning(
            "V2 checkpoint пока отсутствует. "
            "Сначала обучи модель хотя бы "
            "несколько шагов."
        )

    else:

        prompt = st.text_area(
            "Начальный текст:",
            value="Привет! Как дела?",
            height=120,
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            max_new_tokens = st.slider(
                "Новых токенов:",
                min_value=1,
                max_value=256,
                value=64,
            )

        with col2:

            temperature = st.slider(
                "Temperature:",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.1,
            )

        with col3:

            top_k = st.slider(
                "Top-K:",
                min_value=0,
                max_value=200,
                value=50,
                step=5,
            )

        generate_button = st.button(
            "✨ Сгенерировать",
            type="primary",
            width="stretch",
        )

        if generate_button:

            try:

                with st.spinner(
                    "Модель генерирует..."
                ):

                    tokenizer = get_tokenizer()

                    model, checkpoint = (
                        load_model_for_inference()
                    )

                    if model is None:

                        st.error(
                            "Checkpoint не найден."
                        )

                    else:

                        step = (
                            get_checkpoint_step(
                                checkpoint
                            )
                        )

                        result = generate_text(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_k=top_k,
                        )

                        st.subheader(
                            "Результат"
                        )

                        st.write(
                            result
                        )

                        st.caption(
                            "Модель V2, "
                            f"checkpoint step {step}"
                        )

            except Exception as e:

                st.exception(e)


# ============================================================
# MODEL / DIAGNOSTICS
# ============================================================

with tab_model:

    st.header(
        "🧠 MY_AI V2 — состояние проекта"
    )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    st.subheader(
        "Файлы"
    )

    files = [

        (
            "V2 tokenizer",
            TOKENIZER_PATH,
        ),

        (
            "V2 model checkpoint",
            MODEL_SAVE_PATH,
        ),

        (
            "Wiktextract",
            WIKTEXTRACT_PATH,
        ),

        (
            "Cultura local",
            CULTURA_PATH,
        ),

        (
            "Диалоги",
            DIALOGUE_PATH,
        ),
    ]

    for name, path in files:

        if path.exists():

            st.write(
                f"✅ **{name}** — "
                f"`{path.relative_to(BASE_DIR)}` — "
                f"{format_size(path)}"
            )

        else:

            st.write(
                f"❌ **{name}** — "
                "файл не найден"
            )

    st.divider()

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    st.subheader(
        "Архитектура"
    )

    architecture = {

        "Vocabulary":
            VOCAB_SIZE,

        "Embedding":
            EMBED_DIM,

        "Attention heads":
            NUM_HEADS,

        "Transformer layers":
            NUM_LAYERS,

        "FFN":
            f"{EMBED_DIM} → "
            f"{FFN_DIM} → "
            f"{EMBED_DIM}",

        "Context":
            MAX_SEQ_LEN,

        "Micro-batch":
            MICRO_BATCH_SIZE,
    }

    st.json(
        architecture
    )

    model = build_model()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Параметров",
            f"{parameter_count:,}",
        )

    with col2:

        st.metric(
            "Обучаемых",
            f"{trainable_count:,}",
        )

    st.divider()

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    st.subheader(
        "Checkpoint"
    )

    if MODEL_SAVE_PATH.exists():

        try:

            checkpoint = load_checkpoint(
                MODEL_SAVE_PATH
            )

            step = (
                get_checkpoint_step(
                    checkpoint
                )
            )

            loss = (
                get_checkpoint_loss(
                    checkpoint
                )
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Текущий step",
                    step,
                )

            with col2:

                st.metric(
                    "Последний loss",
                    (
                        f"{loss:.4f}"
                        if loss is not None
                        else "—"
                    ),
                )

            st.caption(
                "Checkpoint V2 совместим "
                "с model_state_dict и "
                "старым ключом state_dict."
            )

        except Exception as e:

            st.error(
                f"Ошибка чтения checkpoint: {e}"
            )

    else:

        st.info(
            "Checkpoint ещё не создан."
        )

    st.divider()

    # --------------------------------------------------------
    # Tokenizer test
    # --------------------------------------------------------

    st.subheader(
        "Тест V2 токенизатора"
    )

    test_text = st.text_input(
        "Текст для проверки:",
        value="Привет! Как дела?",
    )

    if st.button(
        "Проверить токенизатор",
        width="stretch",
    ):

        try:

            tokenizer = get_tokenizer()

            ids = tokenizer.encode(
                test_text
            )

            decoded = tokenizer.decode(
                ids
            )

            st.write(
                "**Токены:**",
                tokenizer.tokenizer.encode(
                    test_text
                ).tokens,
            )

            st.write(
                "**ID:**",
                ids,
            )

            st.write(
                "**Decode:**",
                decoded,
            )

            if decoded == test_text:

                st.success(
                    "MATCH: True"
                )

            else:

                st.warning(
                    "MATCH: False"
                )

        except Exception as e:

            st.exception(e)

    st.divider()

    # --------------------------------------------------------
    # Legacy
    # --------------------------------------------------------

    st.subheader(
        "Legacy"
    )

    st.write(
        "Старые файлы пока оставлены "
        "на месте. Этот V2 app их "
        "не использует."
    )

    if LEGACY_TOKENIZER_PATH.exists():

        st.write(
            "Старый tokenizer: "
            f"`{LEGACY_TOKENIZER_PATH.relative_to(BASE_DIR)}`"
        )

    if LEGACY_MODEL_PATH.exists():

        st.write(
            "Старая модель: "
            f"`{LEGACY_MODEL_PATH.relative_to(BASE_DIR)}`"
        )

    st.info(
        "После проверки V2 старые файлы "
        "можно перенести в "
        "archive/legacy, ничего не удаляя."
    )