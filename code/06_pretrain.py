"""
06. 事前学習パイプラインの完全版（05章 5.3 節に対応）
======================================================================
    python 06_pretrain.py                     # --demo と同じ（既定）
    python 06_pretrain.py --demo              # 人工コーパスで縮小版を学習（CPU で数分）
    python 06_pretrain.py --demo --steps 300  # さらに短く（動作確認用）
    python 06_pretrain.py --full              # 日本語 Wikipedia で 215M を学習（GPU 必須）

05章の本文 5.3 節に載っているコード断片（corpus.jsonl の作成 → トークナイザ訓練
→ PretrainDataset → 学習ループ → 保存 → 生成）を、**1本の動くスクリプト**に
まとめたものです。クラス名・変数名は原則本文と揃えてあります。
（例外: PretrainDataset の引数は、--demo / --full の両モードで同じコードを
 使い回すため (data_path, encode_fn, eos_id, max_length) に一般化しています。
 本文版 (data_path, tokenizer, max_length) と中身のやることは同じです）

2つのモード
----------------------------------------------------------------------
--demo（既定）:
    _corpus.py の人工日本語コーパス + _bpe.py の自作 BPE を使う縮小版。
    外部ダウンロード不要・追加ライブラリ不要（PyTorch のみ）で、
    CPU でも数分で「事前学習の1周」を体験できます。
    文法規則が既知のコーパスなので、学習結果を機械採点できます。

--full:
    05章 5.3.1 の手順どおり、日本語 Wikipedia をストリーミングで取得し、
    Hugging Face tokenizers で BPE（語彙 6144）を訓練し、
    215M パラメータ構成（dim=1024, n_layers=18）を事前学習します。
    要 GPU（VRAM 12GB 目安）、要 `pip install datasets tokenizers`。
    既定の 5 万記事（約 0.1B トークン）で一晩程度が目安です。

対応ドキュメント: ../05-自分でLLMを作る.md（5.3 節）
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from _console import title, sub

torch.manual_seed(0)


# ======================================================================
# 1. モデル定義（05章 5.1 節の本文コードそのまま）
# ======================================================================
@dataclass
class ModelConfig:
    dim: int = 768                       # 隠れ層の次元（d_model）
    n_layers: int = 12                   # Decoder 層の数
    n_heads: int = 16                    # Attention のヘッド数
    n_kv_heads: Optional[int] = 8        # GQA 用：K/V のヘッド数（None なら MHA）
    vocab_size: int = 6144               # 語彙数
    hidden_dim: Optional[int] = None     # FFN の中間次元（None なら自動計算）
    multiple_of: int = 64                # hidden_dim を 64 の倍数に揃える
    norm_eps: float = 1e-5               # 正規化のゼロ除算防止用
    max_seq_len: int = 512               # 最大系列長
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # 学習可能なスケール γ

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """回転角を事前計算する（複素数表現）"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)                       # 位置 0, 1, 2, ...
    freqs = torch.outer(t, freqs).float()       # (位置, 次元/2)
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis, x):
    """freqs_cis (T, hd/2) を x (B, T, H, hd/2) に掛けられる形 (1, T, 1, hd/2) にする。

    ★ 05章 5.1.3 の警告どおり、ここを省くと n_heads と次元が衝突して必ず落ちます。
    """
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), \
        f"freqs_cis {tuple(freqs_cis.shape)} が x {tuple(x.shape)} と合いません"
    shape = [d if i in (1, x.ndim - 1) else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    """Q, K に回転を適用する。xq, xk: (バッチ, 系列長, ヘッド数, head_dim)"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x, n_rep: int):
    """GQA: K/V ヘッドを Q ヘッド数に合わせて複製する"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim))


class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_rep = self.n_local_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)      # ① RoPE
        xk = repeat_kv(xk, self.n_rep)                    # ② GQA
        xv = repeat_kv(xv, self.n_rep)
        xq, xk, xv = (t.transpose(1, 2) for t in (xq, xk, xv))

        # ③ Flash Attention（causal=True で自動的に未来をマスク）
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)          # SwiGLU 用に 2/3 に縮める
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(i, args) for i in range(args.n_layers)]
        )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight   # 重み共有（Weight Tying）

        # RoPE テーブルは max_seq_len の 8 倍まで確保（05章 5.1.3 の警告参照）
        freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 8)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self.apply(self._init_weights)
        # 残差経路の出口は層数に応じて小さく初期化（学習の安定化。GPT-2 以来の定石）
        for name, p in self.named_parameters():
            if name.endswith("w2.weight") or name.endswith("wo.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * args.n_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, tokens, targets=None):
        _bsz, seqlen = tokens.shape
        h = self.dropout(self.tok_embeddings(tokens))
        freqs_cis = self.freqs_cis[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cis)
        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
            return logits, loss
        logits = self.output(h[:, [-1], :])
        return logits, None

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=50):
    """05章 5.4 節のサンプリング付き生成（KV キャッシュなしの素朴な版）"""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.args.max_seq_len \
                       else idx[:, -model.args.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ======================================================================
# 2. データの準備（05章 5.3.1）: corpus.jsonl を作る
# ======================================================================
def make_corpus_demo(path: str, n_sentences: int = 20000):
    """_corpus.py の人工日本語文を「1行1文書」の JSONL にする（外部DL不要）。"""
    import _corpus
    sentences = _corpus.make_sentences(n_sentences, seed=0)
    with open(path, "w", encoding="utf-8") as f:
        # 50 文ずつまとめて 1 文書とする（Wikipedia の「記事」に相当する単位）
        for i in range(0, len(sentences), 50):
            doc = "\n".join(sentences[i:i + 50])
            f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
    print(f"  corpus.jsonl を作成: {n_sentences} 文 -> {path}")


def make_corpus_full(path: str, n_docs: int):
    """日本語 Wikipedia をストリーミング取得して JSONL にする（05章 5.3.1 と同じ）。"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "\n  --full には datasets ライブラリが必要です:\n"
            "      pip install datasets\n")

    print(f"  wikimedia/wikipedia (20231101.ja) を先頭 {n_docs} 記事だけ取得します")
    print("  （streaming=True なので全量ダウンロードは発生しません）")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ja",
                      split="train", streaming=True)

    kept = 0
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if kept >= n_docs:
                break
            text = row["text"].strip()
            if len(text) >= 200:            # リダイレクト等の極端に短い記事は除外
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                kept += 1
            if kept and kept % 5000 == 0:
                print(f"    {kept}/{n_docs} 記事 ...")
    print(f"  corpus.jsonl を作成: {kept} 記事 -> {path}")


# ======================================================================
# 3. トークナイザの訓練（05章 5.2.3）
# ======================================================================
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"]


def train_tokenizer_demo(corpus_path: str, vocab_size: int = 400):
    """--demo: 自作 BPE（_bpe.py、02.6章で作ったもの）を再利用する。"""
    from _bpe import ByteLevelBPE
    text = "\n".join(json.loads(line)["text"]
                     for line in open(corpus_path, encoding="utf-8"))
    tok = ByteLevelBPE()
    tok.train(text, vocab_size=vocab_size)
    print(f"  自作 BPE を訓練: 語彙 {len(tok)}")
    # 統一インターフェース: (encode 関数, decode 関数, 語彙数, 文書区切り </s> の id)
    return tok.encode, tok.decode, len(tok), tok.special_to_id["</s>"]


def train_tokenizer_full(corpus_path: str, out_dir: str, vocab_size: int = 6144):
    """--full: Hugging Face tokenizers で BPE を訓練する（05章 5.2.3 の本文コード）。"""
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    except ImportError:
        raise SystemExit(
            "\n  --full には tokenizers ライブラリが必要です:\n"
            "      pip install tokenizers\n")

    tok_path = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        print(f"  既存のトークナイザを再利用: {tok_path}")
        tokenizer = Tokenizer.from_file(tok_path)
    else:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        def read_texts(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)["text"]

        print(f"  BPE トークナイザを訓練中（語彙 {vocab_size}）...")
        tokenizer.train_from_iterator(read_texts(corpus_path), trainer=trainer)
        os.makedirs(out_dir, exist_ok=True)
        tokenizer.save(tok_path)
        print(f"  保存: {tok_path}")

    encode = lambda text: tokenizer.encode(text).ids          # noqa: E731
    decode = lambda ids: tokenizer.decode(ids)                # noqa: E731
    return encode, decode, tokenizer.get_vocab_size(), tokenizer.token_to_id("</s>")


# ======================================================================
# 4. Dataset（05章 5.3.2 の PretrainDataset。load_and_tokenize の中身がこれ）
# ======================================================================
class PretrainDataset(Dataset):
    def __init__(self, data_path, encode_fn, eos_id, max_length=512):
        self.max_length = max_length
        self.encode_fn = encode_fn
        self.eos_id = eos_id
        # 全テキストをトークン化して1本の長い列に連結
        self.data = self.load_and_tokenize(data_path)

    def load_and_tokenize(self, path):
        ids = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                ids.extend(self.encode_fn(json.loads(line)["text"]))
                ids.append(self.eos_id)     # 文書の境界に </s> を挟む
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return (len(self.data) - 1) // self.max_length

    def __getitem__(self, index):
        start = index * self.max_length
        chunk = self.data[start: start + self.max_length + 1]
        X = chunk[:-1].clone()      # 入力
        Y = chunk[1:].clone()       # 正解（1つずらし）＝ 次トークン予測
        return X, Y


# ======================================================================
# 5. 学習ループ（05章 5.3.3 の本文コード。EPOCHS 等の定義もここにある）
# ======================================================================
def get_lr(step, total_steps, warmup_steps, lr_max, lr_min=0.0):
    """学習率スケジューラ：ウォームアップ + コサイン減衰"""
    if step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps              # 線形に上げる
    ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * min(ratio, 1.0)))


def train(model, train_ds, val_ds, device, *, lr_max, batch_size, accum_steps,
          epochs, max_steps=None, ckpt_path="pretrain.pth", log_every=None):
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.1,
                                  betas=(0.9, 0.95))

    ACCUM_STEPS = accum_steps
    EPOCHS = epochs
    TOTAL_STEPS = EPOCHS * (len(loader) // ACCUM_STEPS)
    if max_steps is not None:
        TOTAL_STEPS = min(TOTAL_STEPS, max_steps)
    WARMUP = max(1, int(TOTAL_STEPS * 0.03))     # 全体の 3% をウォームアップに

    # 混合精度（AMP）は CUDA のときだけ使う。CPU では素の fp32 で回す。
    use_amp = (device == "cuda")
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        autocast = lambda: torch.amp.autocast("cuda", dtype=amp_dtype)   # noqa: E731
        # GradScaler が必要なのは fp16 のときだけ（bf16 は表現範囲が広いので不要）
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        print(f"  混合精度: {amp_dtype} (GradScaler {'有効' if scaler.is_enabled() else '不要'})")
    else:
        autocast = nullcontext
        scaler = torch.amp.GradScaler("cpu", enabled=False)   # 何もしない素通し

    print(f"  総ステップ数: {TOTAL_STEPS} (エポック {EPOCHS} × バッチ {len(loader)}"
          f" ÷ 勾配累積 {ACCUM_STEPS}) / ウォームアップ: {WARMUP}")
    print(f"\n  {'step':>6} {'train':>8} {'val':>8} {'lr':>10} {'経過':>8}")

    log_every = log_every or max(1, TOTAL_STEPS // 10)
    t0 = time.perf_counter()
    step = 0
    done = False
    for epoch in range(EPOCHS):
        for i, (X, Y) in enumerate(loader):
            X, Y = X.to(device), Y.to(device)

            # 学習率を更新
            lr = get_lr(step, TOTAL_STEPS, WARMUP, lr_max)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            with autocast():
                _, loss = model(X, targets=Y)
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 勾配クリッピング
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if step % log_every == 0 or step == TOTAL_STEPS - 1:
                    model.eval()
                    with torch.no_grad():
                        vX, vY = next(iter(val_loader))
                        _, vloss = model(vX.to(device), vY.to(device))
                    model.train()
                    print(f"  {step:6d} {loss.item() * ACCUM_STEPS:8.4f} "
                          f"{vloss.item():8.4f} {lr:10.2e} "
                          f"{time.perf_counter() - t0:7.1f}s")

                step += 1
                if step >= TOTAL_STEPS:
                    done = True
                    break
        if done:
            break

    # 学習が終わったら重みを保存（05章 5.3.4 の SFT はこのファイルから始まる）
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  チェックポイントを保存: {ckpt_path}")
    return model


# ======================================================================
# 6. モード別のメイン処理
# ======================================================================
def run_demo(args):
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    title("事前学習パイプライン（縮小デモ版）")
    print("""  05章 5.3 節の一連の流れを、人工コーパスで最初から最後まで通します。
    ① corpus.jsonl 作成 → ② BPE 訓練 → ③ Dataset → ④ 学習 → ⑤ 保存 → ⑥ 生成""")

    sub("① データの準備（5.3.1）")
    corpus_path = "corpus_demo.jsonl"
    make_corpus_demo(corpus_path)

    sub("② トークナイザの訓練（5.2.3）")
    encode, decode, vocab_size, eos_id = train_tokenizer_demo(corpus_path)

    sub("③ Dataset の構築（5.3.2）")
    config = ModelConfig(vocab_size=vocab_size, dim=256, n_layers=4,
                         n_heads=8, n_kv_heads=2, max_seq_len=64)
    dataset = PretrainDataset(corpus_path, encode, eos_id, max_length=config.max_seq_len)
    n_val = max(1, len(dataset) // 10)
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(0))
    print(f"  総トークン数: {len(dataset.data):,} / チャンク数: {len(dataset)}"
          f" (学習 {len(train_ds)} + 検証 {len(val_ds)})")

    sub("④ 学習（5.3.3）")
    model = Transformer(config).to(device)
    print(f"  デバイス: {device} / パラメータ: {model.num_params():,}")
    # チャンク数が少ないのでエポックを重ねて steps 回まで学習する
    epochs = math.ceil(args.steps / max(1, len(train_ds) // args.batch_size))
    train(model, train_ds, val_ds, device,
          lr_max=3e-3, batch_size=args.batch_size, accum_steps=1,
          epochs=epochs, max_steps=args.steps, ckpt_path="pretrain_demo.pth")

    sub("⑤ 保存した重みを読み直せることの確認（5.3.4 の SFT の入口）")
    model2 = Transformer(config).to(device)
    model2.load_state_dict(torch.load("pretrain_demo.pth", map_location=device))
    print("  load_state_dict OK — SFT はこの状態から Dataset を差し替えるだけ")

    sub("⑥ 生成（5.4）")
    import _corpus
    bos = encode("\n")
    out = generate(model2, torch.tensor([bos] * 6, device=device),
                   max_new_tokens=200, temperature=0.9, top_k=30)
    txt = "\n".join(decode(r.tolist()) for r in out)
    for line in txt.split("\n")[1:7]:
        if line.strip():
            print(f"    {line}")
    ok, total, rate = _corpus.score(txt)
    print(f"\n  文法正答率: {ok}/{total} = {rate:.1%}  "
          f"{'[合格]' if rate >= 0.9 else '[要調整: --steps を増やしてください]'}")
    print("""
  ★ これが「事前学習の1周」の全てです。--full にすると
    データが Wikipedia に、語彙が 6144 に、モデルが 215M になるだけで、
    コードの流れは 1 行も変わりません。""")


def run_full(args):
    if not torch.cuda.is_available():
        raise SystemExit(
            "\n  --full は GPU（CUDA）必須です（215M モデルの学習は CPU では非現実的）。\n"
            "  GPU が無い場合は --demo で流れを体験するか、Google Colab をご利用ください。\n")
    device = "cuda"

    title("事前学習パイプライン（215M フル版）")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({vram:.0f} GB)")
    if vram < 11:
        print("  ⚠ VRAM 12GB 未満です。メモリ不足になったら --batch-size を下げてください")

    sub("① データの準備（5.3.1）: 日本語 Wikipedia をストリーミング取得")
    corpus_path = "corpus.jsonl"
    if os.path.exists(corpus_path):
        print(f"  既存の {corpus_path} を再利用します（作り直すなら削除してください）")
    else:
        make_corpus_full(corpus_path, n_docs=args.docs)

    sub("② トークナイザの訓練（5.2.3）: BPE 語彙 6144")
    encode, decode, vocab_size, eos_id = train_tokenizer_full(
        corpus_path, out_dir="my_tokenizer", vocab_size=6144)

    sub("③ Dataset の構築（5.3.2）")
    config = ModelConfig(dim=1024, n_layers=18, n_heads=16, n_kv_heads=8,
                         vocab_size=vocab_size, max_seq_len=512)
    print("  全記事をトークン化しています（数分かかります）...")
    dataset = PretrainDataset(corpus_path, encode, eos_id, max_length=config.max_seq_len)
    n_val = max(1, len(dataset) // 100)
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(0))
    print(f"  総トークン数: {len(dataset.data):,} / チャンク数: {len(dataset)}")

    sub("④ 学習（5.3.3）")
    model = Transformer(config).to(device)
    print(f"  パラメータ: {model.num_params():,} （≈ 215M）")
    train(model, train_ds, val_ds, device,
          lr_max=3e-4, batch_size=args.batch_size, accum_steps=8,
          epochs=1, max_steps=args.steps if args.steps > 0 else None,
          ckpt_path="pretrain.pth", log_every=50)

    sub("⑤ 生成（5.4）")
    prompt_ids = encode("日本の首都は")
    out = generate(model, torch.tensor([prompt_ids], device=device),
                   max_new_tokens=100, temperature=0.8, top_k=50)
    print("  " + decode(out[0].tolist())[:400])
    print("""
  ★ 0.1B トークン程度では、文法的に整った日本語が出始める段階までです。
    ちゃんとした文章にしたければデータを増やしてください
    （Chinchilla 則の目安: 215M × 20 ≈ 4.3B トークン）。
    次は 5.3.4 のとおり、この pretrain.pth を読み込んで SFT へ進みます。""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="05章 5.3 節「小さな LLM を事前学習する」の完全版")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true",
                      help="人工コーパスで縮小版を学習（既定。CPU で数分、追加ライブラリ不要）")
    mode.add_argument("--full", action="store_true",
                      help="日本語 Wikipedia で 215M を学習（GPU 必須、要 datasets/tokenizers）")
    ap.add_argument("--steps", type=int, default=1000,
                    help="学習ステップ数（--full では 0 で全データ 1 エポック）")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--docs", type=int, default=50000,
                    help="--full でダウンロードする Wikipedia 記事数")
    ap.add_argument("--cpu", action="store_true", help="GPU があっても CPU で実行する")
    args = ap.parse_args()

    if args.full:
        if args.batch_size is None:
            args.batch_size = 16
        if args.steps == 1000:      # --full で未指定なら全データを回す
            args.steps = 0
        run_full(args)
    else:
        if args.batch_size is None:
            args.batch_size = 32
        run_demo(args)
