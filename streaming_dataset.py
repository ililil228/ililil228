import torch
from torch.utils.data import IterableDataset


class StreamingWiktextractDataset(IterableDataset):

    def __init__(
        self,
        streamer,
        tokenizer,
        block_size=64
    ):
        super().__init__()

        self.streamer = streamer
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):

        buffer = []

        for text in self.streamer:

            if not text or not text.strip():
                continue

            try:
                ids = self.tokenizer.encode(text)
            except Exception:
                continue

            if not ids:
                continue

            buffer.extend(ids)

            # Нам нужны пары:
            #
            # x = [1, 2, 3, 4]
            # y = [2, 3, 4, 5]
            #
            # То есть модель учится предсказывать
            # следующий токен.

            while len(buffer) >= self.block_size + 1:

                x = torch.tensor(
                    buffer[:self.block_size],
                    dtype=torch.long
                )

                y = torch.tensor(
                    buffer[1:self.block_size + 1],
                    dtype=torch.long
                )

                # Удаляем использованные токены.
                buffer = buffer[self.block_size:]

                yield x, y