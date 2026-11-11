import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.decoders import WordPiece


class BPETokenizer:
    """
    BPE-токенизатор MY_AI v2.

    Важное отличие от старого tokenizer:
    - слова отделяются через WhitespaceSplit;
    - продолжения BPE-слов получают префикс ##;
    - WordPiece decoder склеивает ## обратно без пробела.

    Пример:

        Привет
        ↓
        При + ##вет
        ↓
        Привет
    """

    SPECIAL_TOKENS = [
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]",
    ]

    def __init__(self, vocab_size=16384):
        self.vocab_size = vocab_size

        self.tokenizer = Tokenizer(
            BPE(
                unk_token="[UNK]",
                continuing_subword_prefix="##",
            )
        )

        # Разделяем текст на слова, сохраняя каждое слово
        # отдельным участком для BPE.
        self.tokenizer.pre_tokenizer = WhitespaceSplit()

        # ##piece будет приклеиваться к предыдущему piece.
        self.tokenizer.decoder = WordPiece(
            prefix="##"
        )

    def train(self, files, save_path="models/bpe_tokenizer_v2.json"):
        """
        Обучение tokenizer на файлах.
        """

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=self.SPECIAL_TOKENS,
            continuing_subword_prefix="##",
        )

        self.tokenizer.train(
            files,
            trainer
        )

        os.makedirs(
            os.path.dirname(save_path) or ".",
            exist_ok=True
        )

        self.tokenizer.save(save_path)

        print()
        print("=" * 60)
        print("TOKENIZER V2 TRAINED")
        print("=" * 60)
        print(f"Vocab: {self.get_vocab_size()}")
        print(f"File:  {save_path}")
        print("=" * 60)

    def train_from_iterator(
        self,
        iterator,
        save_path="models/bpe_tokenizer_v2.json"
    ):
        """
        Обучение tokenizer из потокового iterator.
        """

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=self.SPECIAL_TOKENS,
            continuing_subword_prefix="##",
        )

        self.tokenizer.train_from_iterator(
            iterator,
            trainer=trainer
        )

        os.makedirs(
            os.path.dirname(save_path) or ".",
            exist_ok=True
        )

        self.tokenizer.save(save_path)

        print()
        print("=" * 60)
        print("TOKENIZER V2 TRAINED")
        print("=" * 60)
        print(f"Vocab: {self.get_vocab_size()}")
        print(f"File:  {save_path}")
        print("=" * 60)

    def save(self, path="models/bpe_tokenizer_v2.json"):
        """
        Сохранить tokenizer.
        """

        os.makedirs(
            os.path.dirname(path) or ".",
            exist_ok=True
        )

        self.tokenizer.save(path)

    def load(self, path="models/bpe_tokenizer_v2.json"):
        """
        Загрузить готовый tokenizer.
        """

        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text):
        """
        Текст -> token IDs.
        """

        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        """
        Token IDs -> текст.
        """

        return self.tokenizer.decode(ids)

    def encode_batch(self, texts):
        """
        Список текстов -> список списков token IDs.
        """

        return [
            encoded.ids
            for encoded in self.tokenizer.encode_batch(texts)
        ]

    def get_vocab_size(self):
        """
        Размер vocabulary.
        """

        return self.tokenizer.get_vocab_size()

    def get_tokens(self, text):
        """
        Получить pieces вместо ID.
        """

        return self.tokenizer.encode(text).tokens

    def token_to_id(self, token):
        """
        Получить ID конкретного токена.
        """

        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id):
        """
        Получить piece по ID.
        """

        return self.tokenizer.id_to_token(token_id)

    def check(self):
        """
        Базовая проверка tokenizer.
        """

        print()
        print("=" * 60)
        print("TOKENIZER CHECK")
        print("=" * 60)

        print(f"Vocab size: {self.get_vocab_size()}")

        print("\nSpecial tokens:")

        for token in self.SPECIAL_TOKENS:
            token_id = self.token_to_id(token)
            print(f"{token:6} -> {token_id}")

        tests = [
            "Привет! Как дела?",
            "Я изучаю русский язык.",
            "Искусственный интеллект изучает русский язык.",
            "Трансформер использует механизм внимания.",
            "Москва — столица России.",
            "Это простой тест.",
            "Мой компьютер работает на Windows.",
            "Сегодня хорошая погода.",
        ]

        print()

        for text in tests:
            encoded = self.tokenizer.encode(text)

            print("-" * 60)
            print("TEXT:")
            print(text)

            print("\nPIECES:")
            print(" | ".join(
                repr(token)
                for token in encoded.tokens
            ))

            print("\nIDS:")
            print(encoded.ids)

            decoded = self.decode(encoded.ids)

            print("\nDECODE:")
            print(decoded)

            print("\nMATCH:")
            print(text == decoded)

        print()
        print("=" * 60)