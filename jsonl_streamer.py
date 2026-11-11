import json
from pathlib import Path


def stream_wiktextract_text(file_path: str, max_lines: int = None):
    """
    Потоково читает Wiktextract JSONL.
    Извлекает слова, определения и примеры.
    """

    count = 0

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:
            if max_lines is not None and count >= max_lines:
                break

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(data, dict):
                continue

            word = data.get("word")

            if isinstance(word, str):
                word = word.strip()

                if word:
                    yield word
                    count += 1

                    if max_lines is not None and count >= max_lines:
                        break

            senses = data.get("senses", [])

            if not isinstance(senses, list):
                continue

            for sense in senses:
                if not isinstance(sense, dict):
                    continue

                glosses = sense.get("glosses", [])

                if isinstance(glosses, list):
                    for gloss in glosses:

                        if not isinstance(gloss, str):
                            continue

                        gloss = gloss.strip()

                        if not gloss:
                            continue

                        yield gloss
                        count += 1

                        if max_lines is not None and count >= max_lines:
                            break

                if max_lines is not None and count >= max_lines:
                    break

                examples = sense.get("examples", [])

                if isinstance(examples, list):
                    for example in examples:

                        if isinstance(example, str):
                            text = example.strip()

                        elif isinstance(example, dict):
                            text = (
                                example.get("text")
                                or example.get("example")
                                or ""
                            )

                            if not isinstance(text, str):
                                continue

                            text = text.strip()

                        else:
                            continue

                        if not text:
                            continue

                        yield text
                        count += 1

                        if max_lines is not None and count >= max_lines:
                            break

                if max_lines is not None and count >= max_lines:
                    break


def stream_dialogue_jsonl(file_path: str, max_lines: int = None):
    """
    Потоково читает разговорный dataset.jsonl.

    Формат:
    {"sample":["Привет","Как дела?","Нормально"]}

    Реплики одного sample объединяются в один текст.
    """

    count = 0

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:

            if max_lines is not None and count >= max_lines:
                break

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(data, dict):
                continue

            sample = data.get("sample")

            if not isinstance(sample, list):
                continue

            parts = []

            for item in sample:

                if isinstance(item, str):
                    text = item.strip()

                    if text:
                        parts.append(text)

                elif isinstance(item, dict):

                    text = (
                        item.get("text")
                        or item.get("utterance")
                        or item.get("message")
                        or ""
                    )

                    if isinstance(text, str):
                        text = text.strip()

                        if text:
                            parts.append(text)

            if not parts:
                continue

            text = " ".join(parts)

            yield text
            count += 1


class StreamingMYAIDataset:
    """
    Общий поток данных MY_AI.

    Источники:
    - Wiktextract
    - dataset.jsonl

    Оба файла читаются потоково.
    """

    def __init__(
        self,
        wiktextract_path: str = None,
        dialogue_path: str = None,
        wiktextract_max_lines: int = None,
        dialogue_max_lines: int = None,
    ):
        self.wiktextract_path = wiktextract_path
        self.dialogue_path = dialogue_path
        self.wiktextract_max_lines = wiktextract_max_lines
        self.dialogue_max_lines = dialogue_max_lines

    def __iter__(self):

        if self.wiktextract_path:

            path = Path(self.wiktextract_path)

            if path.exists():
                yield from stream_wiktextract_text(
                    str(path),
                    self.wiktextract_max_lines,
                )

        if self.dialogue_path:

            path = Path(self.dialogue_path)

            if path.exists():
                yield from stream_dialogue_jsonl(
                    str(path),
                    self.dialogue_max_lines,
                )


class StreamingWiktextractDataset:
    """
    Старый интерфейс.
    Оставлен для совместимости с существующим app.py.
    """

    def __init__(
        self,
        file_path: str,
        max_lines: int = None,
    ):
        self.file_path = file_path
        self.max_lines = max_lines

    def __iter__(self):
        return stream_wiktextract_text(
            self.file_path,
            self.max_lines,
        )


def test_streamer(file_path: str, count: int = 10):

    print("=" * 60)
    print("WIKTEXTRACT STREAMER TEST")
    print("=" * 60)

    found = 0

    for text in stream_wiktextract_text(
        file_path,
        max_lines=count,
    ):
        print(f"{found + 1}: {text}")
        found += 1

    print("=" * 60)
    print(f"Получено элементов: {found}")
    print("=" * 60)


def test_dialogue_streamer(file_path: str, count: int = 10):

    print("=" * 60)
    print("DIALOGUE JSONL STREAMER TEST")
    print("=" * 60)

    found = 0

    for text in stream_dialogue_jsonl(
        file_path,
        max_lines=count,
    ):
        print(f"{found + 1}: {text}")
        found += 1

    print("=" * 60)
    print(f"Получено диалогов: {found}")
    print("=" * 60)


if __name__ == "__main__":

    wiktextract_file = "resources/raw-wiktextract-data.jsonl"
    dialogue_file = "resources/dataset.jsonl"

    print()
    print("=" * 60)
    print("MY_AI STREAMER TEST")
    print("=" * 60)
    print()

    if Path(wiktextract_file).exists():
        test_streamer(
            wiktextract_file,
            count=10,
        )
    else:
        print(
            f"Wiktextract не найден: {wiktextract_file}"
        )

    print()

    if Path(dialogue_file).exists():
        test_dialogue_streamer(
            dialogue_file,
            count=10,
        )
    else:
        print(
            f"dataset.jsonl не найден: {dialogue_file}"
        )

    print()