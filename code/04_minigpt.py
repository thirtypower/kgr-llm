"""
04. GPT をゼロから作って、実際に学習させる
======================================================================
    python 04_minigpt.py                  # 学習して生成まで（CPU で 3〜5 分）
    python 04_minigpt.py --steps 3000     # もっと学習させる
    python 04_minigpt.py --sanity         # 実装が正しいかの検査だけ実行

これが「自分で LLM を作る」の本体です。
GPT-2 と同じ構造を、外部ライブラリなし（torch だけ）で全部書きます。
規模が違うだけで、GPT-4 も Claude も構造はこれと同じです。

    PART A  モデルを定義する（約 120 行）
    PART B  実装が正しいかを 6 項目で検査する
    PART C  データを用意する
    PART D  学習する
    PART E  生成する・採点する

対応ドキュメント: ../02.7-ミニGPTを作る.md
"""

import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import _corpus
from _bpe import ByteLevelBPE
from _console import title, sub

torch.manual_seed(1337)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# PART A  モデル定義
# ======================================================================
@dataclass
class GPTConfig:
    vocab_size: int = 400        # 語彙数（トークナイザで決まる）
    block_size: int = 64         # 一度に見られる最大トークン数（＝文脈長）
    n_layer: int = 4             # Transformer ブロックの数
    n_head: int = 4              # Attention のヘッド数
    n_embd: int = 128            # 隠れ層の次元 d_model
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """因果的自己注意（Masked Multi-Head Self-Attention）。

    入力 (B, T, C) -> 出力 (B, T, C)。形は変わりません。
    「各トークンのベクトルに、自分より前のトークンの情報を混ぜ込む」層です。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd は n_head で割り切れる必要があります"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

        # Q, K, V を作る線形層。3 つ別々に作ってもよいが、
        # まとめて 1 回の行列積にした方が GPU では速い（実務でも定番）。
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)   # 出力射影 W_O
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # 因果マスク（下三角が True = 見てよい）。
        # register_buffer = 「学習しないがモデルと一緒に保存・GPU移動される変数」
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape                       # batch, 系列長, 次元

        # --- Q, K, V を作る --------------------------------------------
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)      # 各 (B, T, C)

        # --- ヘッドに分割: (B, T, C) -> (B, n_head, T, head_dim) --------
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # --- KV キャッシュ（推論高速化。詳細は 05_llama.py）-------------
        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)     # 過去の K を再利用
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        # --- Attention 本体 -------------------------------------------
        Tq, Tk = q.size(2), k.size(2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B, nh, Tq, Tk)
        # マスクは「Query の位置 i が Key の位置 j を見てよいか」。
        # キャッシュ利用時は Query が末尾 Tq 個だけなので、行を後ろから取る。
        att = att.masked_fill(~self.mask[:, :, Tk - Tq:Tk, :Tk], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                                                   # (B, nh, Tq, hd)

        # --- ヘッドを結合して元の形に戻す ------------------------------
        y = y.transpose(1, 2).contiguous().view(B, Tq, C)
        y = self.resid_dropout(self.proj(y))
        return (y, new_cache) if kv_cache is not None else y


class MLP(nn.Module):
    """位置ごとに独立に働く 2 層の全結合（Feed-Forward Network）。

    Attention が「他のトークンから情報を集める」層なのに対し、
    MLP は「集めた情報を加工する / 知識を引き出す」層です。
    パラメータの約 2/3 はここにあります。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)   # 4 倍に広げるのが慣例
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Transformer ブロック 1 個 = Attention + MLP（どちらも Pre-Norm + 残差）。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, kv_cache=None):
        # ★ x + f(norm(x)) の形（Pre-Norm）。x そのものが一切加工されずに
        #   出口まで通る「高速道路」があることが、深いモデルを学習可能にします。
        if kv_cache is not None:
            a, new_cache = self.attn(self.ln1(x), kv_cache)
            x = x + a
            x = x + self.mlp(self.ln2(x))
            return x, new_cache
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # トークン -> ベクトル
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # 位置 -> ベクトル
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)                      # 最終正規化
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # -> 語彙スコア

        # 重み共有（Weight Tying）: 入口の Embedding と出口の分類器で同じ行列を使う
        self.tok_emb.weight = self.head.weight

        self.apply(self._init_weights)
        # 残差の出口だけ初期化を小さくする（GPT-2 の工夫。深い層でも爆発しない）
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """idx: (B, T) のトークン ID -> logits: (B, T, vocab_size)"""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"系列長 {T} が block_size を超えています"

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))   # (B,T,C) + (T,C) → 自動broadcast
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                                  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1),
                                   ignore_index=-100)
        return logits, loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """1 トークンずつ自己回帰的に生成する。

        idx: (B, T0) の入力（プロンプト）。戻り値: (B, T0 + max_new_tokens)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 文脈長を超えたら古い方を捨てる（＝スライディングウィンドウ）
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                # ★ 最後の位置だけが「次の予測」

            if temperature <= 0:                     # 貪欲法（greedy）
                idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        self.train()
        return idx

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n


# ======================================================================
# PART B  実装が正しいかを検査する
# ======================================================================
def sanity_checks(cfg: GPTConfig):
    title("PART B  実装の自己検査（バグは学習前に潰す）")
    model = MiniGPT(cfg).to(DEVICE)
    ok_all = True

    def report(name, ok, detail=""):
        nonlocal ok_all
        ok_all &= ok
        print(f"  [{'OK' if ok else 'NG'}] {name:38s} {detail}")

    # --- 1. 出力の形 ---------------------------------------------------
    x = torch.randint(0, cfg.vocab_size, (2, 16), device=DEVICE)
    logits, _ = model(x)
    report("出力の形が (B, T, vocab_size)",
           tuple(logits.shape) == (2, 16, cfg.vocab_size), str(tuple(logits.shape)))

    # --- 2. 初期損失 ≈ ln(vocab_size) ---------------------------------
    y = torch.randint(0, cfg.vocab_size, (2, 16), device=DEVICE)
    _, loss = model(x, y)
    expected = math.log(cfg.vocab_size)
    report("初期 loss ≈ ln(語彙数)", abs(loss.item() - expected) < 0.3,
           f"loss={loss.item():.3f}, ln(V)={expected:.3f}")

    # --- 3. 因果性（未来を見ていないか）--------------------------------
    model.eval()
    a = torch.randint(0, cfg.vocab_size, (1, 20), device=DEVICE)
    b = a.clone()
    b[0, 10:] = torch.randint(0, cfg.vocab_size, (10,), device=DEVICE)
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    diff = (la[0, :10] - lb[0, :10]).abs().max().item()
    report("未来のトークンが過去に影響しない", diff < 1e-4, f"最大差={diff:.2e}")

    # --- 4. 全パラメータに勾配が流れるか -------------------------------
    model.train()
    _, loss = model(x, y)
    model.zero_grad()
    loss.backward()
    dead = [n for n, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    report("全パラメータに勾配が届く", not dead, f"届かない: {dead[:3]}" if dead else "")

    # --- 5. KV キャッシュが通常の forward と一致するか ------------------
    model.eval()
    seq = torch.randint(0, cfg.vocab_size, (1, 12), device=DEVICE)
    with torch.no_grad():
        full, _ = model(seq)
        # 1 トークンずつ入れて、キャッシュを使って進める
        caches = [(None, None)] * cfg.n_layer
        outs = []
        for t in range(seq.size(1)):
            tok = seq[:, t:t + 1]
            pos = torch.tensor([t], device=DEVICE)
            h = model.tok_emb(tok) + model.pos_emb(pos)
            for i, blk in enumerate(model.blocks):
                h, caches[i] = blk(h, caches[i])
            outs.append(model.head(model.ln_f(h)))
        incr = torch.cat(outs, dim=1)
    d = (full - incr).abs().max().item()
    report("KV キャッシュ生成 == 一括 forward", d < 1e-3, f"最大差={d:.2e}")

    # --- 6. 1 バッチだけ過学習できるか（学習能力の確認）-----------------
    model = MiniGPT(cfg).to(DEVICE)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    xb = torch.randint(0, cfg.vocab_size, (4, 32), device=DEVICE)
    yb = torch.randint(0, cfg.vocab_size, (4, 32), device=DEVICE)
    for _ in range(200):
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    report("1 バッチを暗記できる（loss→0）", loss.item() < 0.1, f"loss={loss.item():.4f}")

    print(f"""
  ★ 特に重要なのが 2・3・6 です。
     2 が通れば「損失計算と初期化」が正しい。
     3 が通れば「因果マスク」が正しい。
     6 が通れば「勾配が流れて学習できる」ことが確定する。
     この 3 つが通ってから初めて本番データで学習を始めてください。
     順番を守らないと「データが悪いのかコードが悪いのか」で何日も溶かします。

  総合判定: {'すべて OK' if ok_all else '失敗あり'}""")
    return ok_all


# ======================================================================
# PART C  データ
# ======================================================================
def build_data(cfg_vocab: int):
    title("PART C  データとトークナイザ")
    text = _corpus.make_text(n=20000, seed=0)
    print(f"  コーパス      : {len(text):,} 文字")
    print(f"  規則          : {_corpus.describe()}")

    tok = ByteLevelBPE()
    tok.train(text, vocab_size=cfg_vocab)
    print(f"  語彙サイズ    : {len(tok)}")

    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"  総トークン数  : {len(ids):,}")
    print(f"  圧縮率        : {len(text) / len(ids):.2f} 文字/トークン")

    n = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]
    print(f"  学習用        : {len(train_ids):,} トークン")
    print(f"  検証用        : {len(val_ids):,} トークン")

    # --- 理論的に到達できる最小の loss を計算する ----------------------
    # このコーパスは自分で作ったので、含まれる「本当の不確実性」が分かります。
    # 1 文あたり: 文型 5 通り × スロット 6 通り × 6 通り = 180 通りが等確率
    #  → 1 文が持つ情報量 = ln(180) ナット
    # それ以外（助詞や語尾）は文型が決まれば完全に決定的なので情報量 0。
    n_sentences = 20000
    min_loss = n_sentences * math.log(180) / len(ids)
    print(f"""
  ★ このコーパスの理論的な loss の下限 = {min_loss:.4f}
    （1 文の情報量 ln(180)={math.log(180):.2f} ナット ÷ 1 文あたり
      平均 {len(ids) / n_sentences:.1f} トークン）
    学習後の val loss がこの値に近づけば「学べる限界まで学んだ」という意味です。
    それ以上は下がりません（下がったら学習データを暗記しているだけ）。

  ★ 検証用を分ける理由: 学習データの loss はモデルが「暗記」するだけでも
    下がります。未知のデータでの loss（val loss）が下がって初めて
    「規則を学んだ」と言えます。両者が乖離し始めたら過学習の合図です。""")
    return tok, train_ids, val_ids, min_loss


def get_batch(data, block_size, batch_size, device):
    """データ列からランダムな位置を batch_size 個切り出す。

    x = data[i   : i+block_size]
    y = data[i+1 : i+block_size+1]   ← 1 つ後ろにずらしたもの＝正解
    """
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


# ======================================================================
# PART D  学習
# ======================================================================
@torch.no_grad()
def estimate_loss(model, train_ids, val_ids, cfg, batch_size, iters=20):
    model.eval()
    out = {}
    for split, data in [("train", train_ids), ("val", val_ids)]:
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, cfg.block_size, batch_size, DEVICE)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(step, total, warmup, lr_max, lr_min_ratio=0.1):
    """ウォームアップ + コサイン減衰。LLM 学習の事実上の標準。"""
    if step < warmup:
        return lr_max * (step + 1) / warmup
    ratio = (step - warmup) / max(1, total - warmup)
    return lr_max * (lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * ratio)))


def train(model, train_ids, val_ids, cfg, steps=1500, batch_size=32, lr=3e-3,
          min_loss=None):
    title("PART D  学習")
    print(f"  デバイス      : {DEVICE}")
    print(f"  パラメータ数  : {model.num_params():,}")

    sub("パラメータの内訳")
    groups = {}
    for name, p in model.named_parameters():
        key = ("Embedding(トークン)" if "tok_emb" in name else
               "Embedding(位置)" if "pos_emb" in name else
               "Attention" if "attn" in name else
               "MLP" if "mlp" in name else "LayerNorm 等")
        groups[key] = groups.get(key, 0) + p.numel()
    total = sum(groups.values())
    for k, v in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"    {k:22s} {v:>10,}  ({v / total * 100:5.1f}%)")
    print("    ※ 重み共有により tok_emb と出力層は同じ行列（二重計上していません）")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    warmup = max(10, steps // 20)

    sub("学習ログ")
    print(f"  {'step':>6} {'lr':>9} {'train':>8} {'val':>8} {'val ppl':>9} {'経過':>7}")
    t0 = time.time()
    for step in range(steps):
        cur_lr = get_lr(step, steps, warmup, lr)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        x, y = get_batch(train_ids, cfg.block_size, batch_size, DEVICE)
        _, loss = model(x, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # 勾配クリッピング: 勾配ベクトル全体のノルムを 1.0 以下に抑える。
        # 稀に来る巨大勾配で重みが吹き飛ぶ事故を防ぐ、安い保険。
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % max(1, steps // 10) == 0 or step == steps - 1:
            e = estimate_loss(model, train_ids, val_ids, cfg, batch_size)
            print(f"  {step:6d} {cur_lr:9.2e} {e['train']:8.4f} {e['val']:8.4f} "
                  f"{math.exp(min(e['val'], 20)):9.2f} {time.time() - t0:6.1f}s")

    if min_loss is not None:
        gap = e["val"] - min_loss
        print(f"""
  最終 val loss   = {e['val']:.4f}
  理論下限        = {min_loss:.4f}
  差              = {gap:+.4f}
  → 差がほぼ 0 なら、このデータから学べることは全部学んだ状態です。
    LLM の学習で「loss がこれ以上下がらない」のは、多くの場合バグではなく
    「そのデータの持つ情報量を学び切った」ことを意味します。
    そこから先に進むには、データを増やすかモデルを大きくするしかありません。
    これが Scaling Law（規模の法則）が生まれる素朴な理由です。""")
    return model


# ======================================================================
# PART E  生成と採点
# ======================================================================
def generate_and_score(model, tok, cfg):
    title("PART E  生成してみる")

    # 生成の起点。このコーパスは文が改行区切りなので「改行」が文の開始合図。
    # ★ 学習時に一度も見ていないトークン（<pad> など）を起点にしてはいけません。
    #   モデルにとって完全な未知の状態から始まるので出力が壊れます。
    bos = tok.encode("\n")

    sub("温度を変えて生成（プロンプトなし・50 トークン）")
    for temp in [0.0, 0.5, 1.0, 1.5]:
        start = torch.tensor([bos], dtype=torch.long, device=DEVICE)
        out = model.generate(start, 50, temperature=temp, top_k=50)
        text = tok.decode(out[0].tolist()).strip().replace("\n", " / ")
        label = "greedy" if temp == 0 else f"T={temp}"
        print(f"  [{label:>7}] {text[:88]}")

    print("""
  読み方:
    T=0    完全に決定的（greedy）。毎回いちばん確率の高いトークンを選びます。
           → 同じ文を延々と繰り返しがち。「無難だが単調」の正体がこれです。
    T=0.5  やや堅実。実務の要約・抽出・分類タスク向き。
    T=1.0  学習した確率分布そのまま。
    T=1.5  分布を平坦化して冒険させる。多様だが壊れた出力も増える。

  ★ 温度 T は softmax の直前で logits を T で割る操作です。
      T が小さい → 差が拡大 → 一番確率の高いものに集中
      T が大きい → 差が縮小 → 平坦（＝ランダムに近づく）""")

    sub("プロンプトを与えて続きを書かせる")
    for prompt in ["猫が", "太郎は", "公園に"]:
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=DEVICE)
        out = model.generate(ids, 12, temperature=0.8, top_k=20)
        text = tok.decode(out[0].tolist()).split("\n")[0]
        print(f"  「{prompt}」 → 「{text}」")

    sub("文法の正答率で採点する（このコーパスは規則が既知なので採点できる）")
    start = torch.tensor([bos] * 8, dtype=torch.long, device=DEVICE)
    out = model.generate(start, 400, temperature=0.9, top_k=30)
    all_text = "\n".join(tok.decode(row.tolist()) for row in out)
    ok, total, rate = _corpus.score(all_text)
    print(f"  生成した文: {total} 文中 {ok} 文が文法的に正しい  →  正答率 {rate:.1%}")
    print(f"""
  判定:
    90% 以上 : 実装は正しく、規則を学習できています
    50〜90%  : 学習ステップ数かモデルサイズが足りません（--steps を増やす）
    30% 未満 : 実装バグの可能性が高い（PART B の検査に戻ってください）

  今回の結果: {'合格' if rate >= 0.9 else '要調整' if rate >= 0.5 else '要デバッグ'}""")

    sub("次のトークンの確率分布を覗く")
    prompt = "猫が"
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits, _ = model(ids)
    probs = F.softmax(logits[0, -1], dim=-1)
    top = torch.topk(probs, 8)
    print(f"  「{prompt}」の次に来るトークンの予測 上位 8:")
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        piece = tok.vocab[i].decode("utf-8", errors="replace")
        bar = "#" * int(p * 50)
        print(f"    {p:6.1%}  '{piece}'  {bar}")
    print("""
  ★ これが LLM の全てです。モデルは「次のトークンの確率分布」しか出しません。
    それを 1 個サンプリングして末尾に足し、また入力に戻す。この繰り返しだけで
    文章が生まれます。ChatGPT も内部でやっているのは完全にこれと同じです。""")
    return rate


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--vocab", type=int, default=400)
    ap.add_argument("--sanity", action="store_true", help="自己検査だけ実行する")
    args = ap.parse_args()

    cfg = GPTConfig(vocab_size=args.vocab)

    if args.sanity:
        sanity_checks(cfg)
        return

    sanity_checks(cfg)
    tok, train_ids, val_ids, min_loss = build_data(args.vocab)
    cfg.vocab_size = len(tok)

    model = MiniGPT(cfg).to(DEVICE)
    train(model, train_ids, val_ids, cfg, steps=args.steps,
          batch_size=args.batch_size, lr=args.lr, min_loss=min_loss)
    generate_and_score(model, tok, cfg)

    torch.save({"model": model.state_dict(), "config": cfg}, "minigpt.pt")
    title("完了")
    print("""
  モデルを minigpt.pt に保存しました。

  ここまでで作ったもの:
    ・トークナイザ（BPE、自作）
    ・GPT 本体（Attention / MLP / LayerNorm / 残差、自作）
    ・学習ループ（AdamW / ウォームアップ / コサイン減衰 / 勾配クリッピング）
    ・生成（temperature / top-k サンプリング）

  これは GPT-2 の完全なミニチュアです。次は現代版（LLaMA2）へ:
      python 05_llama.py
""")


if __name__ == "__main__":
    main()
