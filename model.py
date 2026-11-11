import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, max_seq_len):
        super().__init__()

        if embed_size % num_heads != 0:
            raise ValueError(
                f"embed_size ({embed_size}) должен делиться на num_heads ({num_heads})"
            )

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.max_seq_len = max_seq_len

        self.qkv = nn.Linear(embed_size, embed_size * 3)
        self.out_proj = nn.Linear(embed_size, embed_size)

        # Маска создаётся один раз.
        # Не используем -inf, потому что DirectML иногда падает на softmax.
        mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len),
            diagonal=1
        )

        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        batch_size, seq_len, embed_size = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Длина последовательности {seq_len} больше max_seq_len={self.max_seq_len}"
            )

        # B, T, 3C
        qkv = self.qkv(x)

        # Разделяем Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)

        # B, T, C -> B, heads, T, head_dim
        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # 1. Attention scores (B, heads, T, T)
        att = torch.matmul(q, k.transpose(-2, -1))

        # 2. ИСПРАВЛЕНО: Сначала применяем scale к сырым результатам умножения
        scale = self.head_dim ** -0.5
        att = att * scale

        # Берём нужный кусок causal mask
        mask = self.causal_mask[:seq_len, :seq_len]

        # 3. ИСПРАВЛЕНО: Маску накладываем строго после масштабирования
        # Теперь -10000.0 будет отрабатывать корректно для Softmax в DirectML
        att = att + mask * -10000.0

        # Softmax
        att = torch.softmax(att, dim=-1)

        # B, heads, T, head_dim
        out = torch.matmul(att, v)

        # B, heads, T, head_dim -> B, T, heads, head_dim
        out = out.transpose(1, 2).contiguous()

        # -> B, T, C
        out = out.view(
            batch_size,
            seq_len,
            embed_size
        )

        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, max_seq_len):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_size)
        self.attn = CausalSelfAttention(
            embed_size=embed_size,
            num_heads=num_heads,
            max_seq_len=max_seq_len
        )

        self.ln2 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size)

    def forward(self, x):
        # Pre-Norm Transformer
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x


class BPEModel(nn.Module):
    def __init__(
        self,
        vocab_size=16384,
        embed_size=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=64
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Токены
        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_size
        )

        # Позиции 0..63
        self.position_embedding = nn.Embedding(
            max_seq_len,
            embed_size
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_size=embed_size,
                num_heads=num_heads,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(embed_size)

        # Выход в vocab
        self.lm_head = nn.Linear(
            embed_size,
            vocab_size,
            bias=False
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, x):
        batch_size, seq_len = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Вход имеет длину {seq_len}, "
                f"но модель поддерживает максимум {self.max_seq_len}"
            )

        # Позиции: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(
            0,
            seq_len,
            device=x.device
        )

        # B,T,C
        token_emb = self.token_embedding(x)

        # T,C
        pos_emb = self.position_embedding(positions)

        # broadcasting: B,T,C + T,C
        x = token_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)

        # B,T,Vocab
        logits = self.lm_head(x)

        return logits, None


if __name__ == "__main__":
    print("=== TEST model.py ===")

    model = BPEModel(
        vocab_size=16384,
        embed_size=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=64
    )

    test_input = torch.randint(
        low=0,
        high=16384,
        size=(2, 64),
        dtype=torch.long
    )

    logits, _ = model(test_input)

    print("Input:", test_input.shape)
    print("Output:", logits.shape)

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Parameters: {params:,}")
    print(f"Trainable:  {trainable:,}")

    expected_shape = (2, 64, 16384)

    if tuple(logits.shape) == expected_shape:
        print("MODEL TEST: OK")
    else:
        print(
            "MODEL TEST: ERROR",
            logits.shape,
            "expected",
            expected_shape
        )
