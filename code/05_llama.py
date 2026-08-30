"""
05. GPT-2 を現代化する: LLaMA2 アーキテクチャ
======================================================================
    python 05_llama.py            # 全部の検証を実行（CPU/GPU どちらでも数十秒）
    python 05_llama.py --train    # ミニコーパスで実際に学習まで行う

04_minigpt.py で作った GPT-2 相当のモデルを、2024〜2026 年の標準形に
差し替えていきます。変更は 5 点だけです。

    LayerNorm         -> RMSNorm            （高速化）
    学習型の位置埋め込み -> RoPE               （相対位置・長文対応）
    MHA               -> GQA                （KV キャッシュのメモリ削減）
    ReLU の FFN       -> SwiGLU             （表現力）
    バイアス項         -> 全部なくす           （不要と分かった）

各技術について「本当にそうなっているか」を数値で確認します。
名前を覚えるのではなく、確認したことだけを信じてください。

対応ドキュメント: ../05-自分でLLMを作る.md
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from _console import title, sub

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
@dataclass
class LlamaConfig:
    dim: int = 256                        # 隠れ次元 d_model
    n_layers: int = 4                     # Decoder 層の数
    n_heads: int = 8                      # Query のヘッド数
    n_kv_heads: Optional[int] = 2         # K/V のヘッド数（None なら MHA と同じ）
    vocab_size: int = 400
    hidden_dim: Optional[int] = None      # FFN の中間次元（None なら自動計算）
    multiple_of: int = 64                 # hidden_dim をこの倍数に揃える
    norm_eps: float = 1e-5
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


# ======================================================================
# 1. RMSNorm
# ======================================================================
class RMSNorm(nn.Module):
    """LayerNorm から「平均を引く」処理を省いたもの。

    LayerNorm: y = g * (x - mean) / sqrt(var + eps) + b
    RMSNorm  : y = g * x / sqrt(mean(x^2) + eps)

    平均の計算とバイアス項が消えるので速く、性能はほぼ変わらないことが
    実験的に確かめられています（Zhang & Sennrich, 2019）。
    LLaMA 以降ほぼ全ての LLM がこちらを使っています。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))    # 学習するスケール g

    def _norm(self, x):
        # rsqrt(z) = 1/sqrt(z)。除算より速いのでこう書くのが慣例。
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # ★ float32 に上げてから正規化する。
        #   bf16 のままだと x^2 の平均で桁が溢れ、学習が突然壊れることがあります。
        return self.weight * self._norm(x.float()).type_as(x)


# ======================================================================
# 2. RoPE（Rotary Position Embedding）
# ======================================================================
def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """各位置・各次元ペアの「回転角」を、複素数の形で事前計算する。

    戻り値: (max_seq_len, head_dim/2) の complex テンソル。
            要素は大きさ 1・偏角 θ の複素数 = 「θ だけ回す」という操作そのもの。
    """
    # 次元ペアごとに回転の速さを変える。
    # 前の方の次元は速く回り（近距離の区別が得意）、
    # 後ろの方の次元はゆっくり回る（遠距離の区別が得意）。
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()               # 位置 0,1,2,...
    freqs = torch.outer(t, freqs)                       # (T, head_dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)   # 極形式 -> 複素数


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """freqs_cis (T, hd/2) を x (B, T, H, hd/2) にブロードキャストできる形にする。

    ★ ここは実装で最も間違えやすい箇所です。
      形を合わせずに掛けると、静かに間違った位置情報が入り、
      「学習は進むのに性能が出ない」という最悪のバグになります。
    """
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), \
        f"freqs_cis {tuple(freqs_cis.shape)} が x {tuple(x.shape)} と合いません"
    shape = [d if i in (1, x.ndim - 1) else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)                       # (1, T, 1, hd/2)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Q と K を位置に応じて回転させる。xq, xk: (B, T, H, head_dim)"""
    # 隣り合う 2 次元を 1 組の複素数とみなす: (..., hd) -> (..., hd/2) complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    fc = reshape_for_broadcast(freqs_cis, xq_)
    # 複素数の掛け算 = 2 次元平面での回転
    xq_out = torch.view_as_real(xq_ * fc).flatten(3)
    xk_out = torch.view_as_real(xk_ * reshape_for_broadcast(freqs_cis, xk_)).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ======================================================================
# 3. GQA（Grouped-Query Attention）
# ======================================================================
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """K/V のヘッドを Q のヘッド数に合わせて複製する。 x: (B, T, n_kv_heads, hd)"""
    if n_rep == 1:
        return x
    B, T, n_kv, hd = x.shape
    return (x[:, :, :, None, :]
            .expand(B, T, n_kv, n_rep, hd)
            .reshape(B, T, n_kv * n_rep, hd))


class Attention(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_heads if cfg.n_kv_heads is None else cfg.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = cfg.head_dim

        # ★ K と V だけ出力次元が小さい。ここが GQA の実体。
        self.wq = nn.Linear(cfg.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout_p = cfg.dropout

    def forward(self, x, freqs_cis, cache=None):
        B, T, _ = x.shape

        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # ① 位置情報を回転で注入（K と Q のみ。V は回さない）
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # ② KV キャッシュ: 過去に計算した K/V をつなげる
        new_cache = None
        if cache is not None:
            past_k, past_v = cache
            if past_k is not None:
                xk = torch.cat([past_k, xk], dim=1)
                xv = torch.cat([past_v, xv], dim=1)
            new_cache = (xk, xv)          # ★ repeat_kv する『前』を保存するのが肝
                                          #   （複製後を保存するとメモリ削減効果が消える）

        # ③ GQA: K/V を Q のヘッド数まで複製
        k = repeat_kv(xk, self.n_rep)
        v = repeat_kv(xv, self.n_rep)

        # (B, T, H, hd) -> (B, H, T, hd)
        xq, k, v = xq.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        Tq, Tk = xq.size(2), k.size(2)
        # 因果マスク: Query の位置 (Tk-Tq+i) が Key の位置 j を見てよいのは j <= それ
        if Tq == 1:
            mask = None                   # 生成時の 1 トークン: 全部の過去を見てよい
        else:
            m = torch.ones(Tq, Tk, dtype=torch.bool, device=x.device).tril(Tk - Tq)
            mask = torch.zeros(Tq, Tk, device=x.device).masked_fill(~m, float("-inf"))

        out = F.scaled_dot_product_attention(
            xq, k, v, attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0)

        out = out.transpose(1, 2).contiguous().view(B, Tq, -1)
        return self.wo(out), new_cache


# ======================================================================
# 4. SwiGLU FFN
# ======================================================================
class FeedForward(nn.Module):
    """SwiGLU: 3 つの線形層 + ゲート機構。

        out = W2( SiLU(W1 x) * W3 x )

    W3 x が「どの情報を通すか」のゲートとして働きます。
    層が 1 つ増える分、中間次元を 2/3 に縮めてパラメータ数を揃えるのが慣例。
    """

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        hidden = cfg.hidden_dim
        if hidden is None:
            hidden = 4 * cfg.dim                     # 従来の FFN と同じ 4 倍から
            hidden = int(2 * hidden / 3)             # 層が 3 つになるので 2/3 に
            # GPU の行列積が効率的になるよう倍数に切り上げる
            hidden = cfg.multiple_of * math.ceil(hidden / cfg.multiple_of)
        self.hidden_dim = hidden
        self.w1 = nn.Linear(cfg.dim, hidden, bias=False)   # ゲートを作る側
        self.w2 = nn.Linear(hidden, cfg.dim, bias=False)   # 出力側
        self.w3 = nn.Linear(cfg.dim, hidden, bias=False)   # 通す中身の側
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ======================================================================
# 5. 組み立て
# ======================================================================
class TransformerBlock(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.attention = Attention(cfg)
        self.feed_forward = FeedForward(cfg)
        self.attention_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)

    def forward(self, x, freqs_cis, cache=None):
        # Pre-Norm + 残差接続
        attn_out, new_cache = self.attention(self.attention_norm(x), freqs_cis, cache)
        h = x + attn_out
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, new_cache


class Llama(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight       # 重み共有

        # RoPE の回転角は学習しないので事前計算してバッファに置く。
        # ★ max_seq_len ちょうどではなく 8 倍まで用意しておく:
        #   RoPE は「学習した長さを超えた位置」にもある程度外挿できるのが利点で、
        #   生成時は学習時の文脈長を超えることが普通にあるため。
        #   （テーブルが足りないと生成の途中で空のスライスを掴んで落ちます）
        self.rope_len = cfg.max_seq_len * 8
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(cfg.head_dim, self.rope_len, cfg.rope_theta),
            persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("w2.weight") or name.endswith("wo.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, tokens, targets=None, caches=None, start_pos: int = 0):
        B, T = tokens.shape
        h = self.dropout(self.tok_embeddings(tokens))
        # ★ 生成時は「今どの絶対位置にいるか」に合わせて回転角を切り出す
        assert start_pos + T <= self.rope_len, (
            f"位置 {start_pos + T} が RoPE テーブル長 {self.rope_len} を超えました。"
            f"生成トークン数を減らすか、rope_len を増やしてください")
        freqs_cis = self.freqs_cis[start_pos:start_pos + T]

        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            h, nc = layer(h, freqs_cis, cache)
            new_caches.append(nc)
        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=-100)
            return logits, loss, new_caches
        # 推論時は最後の位置だけ計算すれば十分（語彙が大きいほど効く）
        logits = self.output(h[:, -1:, :])
        return logits, None, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None,
                 use_cache=True):
        self.eval()
        caches = [(None, None)] * self.cfg.n_layers if use_cache else None
        pos = 0
        cur = idx
        for _ in range(max_new_tokens):
            if use_cache:
                logits, _, caches = self(cur, caches=caches, start_pos=pos)
                pos += cur.size(1)
            else:
                ctx = idx[:, -self.cfg.max_seq_len:]
                logits, _, _ = self(ctx)
            logits = logits[:, -1, :]

            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            cur = nxt if use_cache else idx     # キャッシュ有りなら新トークンだけ渡す
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ======================================================================
# 検証
# ======================================================================
def verify():
    cfg = LlamaConfig()

    # ------------------------------------------------------------------
    title("検証 1  RMSNorm は LayerNorm と同じ働きをして、より速い")
    # ------------------------------------------------------------------
    x = torch.randn(64, 128, 512)
    rms, ln = RMSNorm(512), nn.LayerNorm(512)
    print(f"  入力の各行のスケール（RMS）: 平均 {x.pow(2).mean(-1).sqrt().mean():.4f}")
    print(f"  RMSNorm 後               : 平均 {rms(x).pow(2).mean(-1).sqrt().mean():.4f}  ← 1 に揃った")
    print(f"  LayerNorm 後             : 平均 {ln(x).pow(2).mean(-1).sqrt().mean():.4f}")

    for name, mod in [("LayerNorm", ln), ("RMSNorm  ", rms)]:
        mod(x)                                     # ウォームアップ
        t0 = time.perf_counter()
        for _ in range(100):
            mod(x)
        print(f"  {name} 100 回: {(time.perf_counter() - t0) * 1000:7.1f} ms")
    print("""
  ★ 「平均を引く」処理と bias を消しただけですが、
    正規化は全層・全トークンで走るので、この差が積み上がります。""")

    # ------------------------------------------------------------------
    title("検証 2  RoPE は本当に「相対位置」だけで決まるのか")
    # ------------------------------------------------------------------
    print("""RoPE の主張:
    「Q を位置 m だけ、K を位置 n だけ回転させて内積を取ると、
      結果は m と n の『差』だけで決まる」

  絶対位置を入れているのに相対位置が出てくる、という不思議な性質です。
  本当かどうか、実際に計算して確かめます。""")

    hd = 64
    fc = precompute_freqs_cis(hd, 128)
    q = torch.randn(1, 1, 1, hd)            # 同じベクトルを使い回す
    k = torch.randn(1, 1, 1, hd)

    def dot_at(m, n):
        qm, _ = apply_rotary_emb(q, q, fc[m:m + 1])
        _, kn = apply_rotary_emb(k, k, fc[n:n + 1])
        return (qm * kn).sum().item()

    print(f"\n  {'(m, n)':>12} {'m - n':>7} {'内積':>12}")
    print("  " + "-" * 34)
    for m, n in [(3, 1), (10, 8), (50, 48), (100, 98), (5, 0), (105, 100)]:
        print(f"  {str((m, n)):>12} {m - n:7d} {dot_at(m, n):12.6f}")

    d1, d2, d3 = dot_at(3, 1), dot_at(50, 48), dot_at(100, 98)
    spread = max(d1, d2, d3) - min(d1, d2, d3)
    print(f"\n  差が 2 の 3 組の内積のばらつき = {spread:.2e}")
    assert spread < 1e-4, "RoPE の相対位置性が壊れています"
    print("""  [OK] 位置が 3・50・100 と全く違っても、差が同じなら内積は同一。

  ★ なぜ嬉しいか:
    「2 つ前の単語」という関係を、文の先頭でも 1000 トークン目でも
    同じ形で表現できます。学習した長さより長い文にもある程度対応でき、
    追加の学習パラメータも要りません。だから現代 LLM の標準になりました。""")

    # 距離と内積の減衰
    sub("距離が離れると内積はどうなるか（long-term decay）")
    print(f"  {'距離':>6} {'内積の平均(100 サンプル)':>26}")
    for dist in [0, 1, 2, 4, 8, 16, 32, 64]:
        vals = []
        for _ in range(100):
            a, b = torch.randn(1, 1, 1, hd), torch.randn(1, 1, 1, hd)
            am, _ = apply_rotary_emb(a, a, fc[dist:dist + 1])
            _, bn = apply_rotary_emb(b, b, fc[0:1])
            vals.append((am * bn).sum().item())
        print(f"  {dist:6d} {sum(vals) / len(vals):26.4f}")
    print("""  → 平均は 0 付近ですが、距離が離れるほど「たまたま高い内積」が
    出にくくなります（回転の位相がバラけるため）。
    これが「遠いトークンほど弱く結びつく」という自然な性質を生みます。""")

    # ------------------------------------------------------------------
    title("検証 3  GQA はどれだけメモリを減らすのか")
    # ------------------------------------------------------------------
    print("""KV キャッシュのサイズ = 2(K と V) × 層数 × 系列長 × n_kv_heads × head_dim × バイト数

  生成中、過去の K/V は全部メモリに保持し続けなければなりません。
  長い文脈・大きいバッチではモデル本体より KV キャッシュの方が大きくなります。
  GQA は K/V のヘッド数だけを減らすことで、ここを直接削ります。""")

    def kv_bytes(layers, seq, kv_heads, head_dim, batch=1, bytes_per=2):
        return 2 * layers * batch * seq * kv_heads * head_dim * bytes_per

    print(f"\n  例: 32 層 / head_dim 128 / Q ヘッド 32 / 系列長 4096 / batch 8 / bf16\n")
    print(f"  {'方式':>10} {'KV ヘッド数':>12} {'KV キャッシュ':>16} {'削減率':>8}")
    print("  " + "-" * 50)
    base = kv_bytes(32, 4096, 32, 128, 8)
    for name, kvh in [("MHA", 32), ("GQA(8)", 8), ("GQA(4)", 4), ("MQA", 1)]:
        b = kv_bytes(32, 4096, kvh, 128, 8)
        print(f"  {name:>10} {kvh:12d} {b / 1e9:14.2f} GB {(1 - b / base) * 100:7.0f}%")
    print("""
  ★ MHA だと KV キャッシュだけで 17 GB。GQA(8) なら 4 GB。
    「同じ GPU で 4 倍のリクエストを捌ける」という意味なので、
    推論コストに直結します。LLaMA-2 70B 以降ほぼ全て GQA です。
  ★ 性能低下はごくわずか。K/V は「何を提供するか」なので、
    Q ほど多様性が要らない、というのが直感的な理由です。""")

    m_mha = Llama(LlamaConfig(n_kv_heads=None)).num_params()
    m_gqa = Llama(LlamaConfig(n_kv_heads=2)).num_params()
    print(f"  参考: 本モデルのパラメータ数  MHA {m_mha:,} -> GQA {m_gqa:,} "
          f"({(1 - m_gqa / m_mha) * 100:.1f}% 減)")

    # ------------------------------------------------------------------
    title("検証 4  SwiGLU のゲート機構")
    # ------------------------------------------------------------------
    ff = FeedForward(cfg)
    print(f"  dim={cfg.dim} -> hidden_dim={ff.hidden_dim} -> dim={cfg.dim}")
    print(f"  ({4 * cfg.dim} の 2/3 = {int(4 * cfg.dim * 2 / 3)} を "
          f"{cfg.multiple_of} の倍数に切り上げ)")
    xs = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0])
    print(f"\n  {'x':>6} {'ReLU':>8} {'GELU':>8} {'SiLU':>8}")
    for v in xs:
        print(f"  {v:6.1f} {F.relu(v):8.3f} {F.gelu(v):8.3f} {F.silu(v):8.3f}")
    print("""
  ★ ReLU は負の入力を完全に捨てる（勾配も 0 = そのニューロンは死ぬ）。
    SiLU/GELU は負側もわずかに通すので勾配が流れ続け、学習が安定します。
  ★ ゲート（w3 側）の役割: SiLU(w1 x) が 0 に近い次元は出力が消えます。
    「文脈に応じて、通す情報を選ぶスイッチ」を学習で獲得できます。""")

    # ------------------------------------------------------------------
    title("検証 5  KV キャッシュ: 正しさと速さ")
    # ------------------------------------------------------------------
    print("""生成は 1 トークンずつ進みます。素朴にやると毎回、系列全体を
  計算し直すことになり、t トークン目のコストが O(t^2)。
  でも過去のトークンの K/V は毎回同じ値です（因果マスクのおかげで
  未来の影響を受けないため）。だから保存して使い回せます。

  ただし「使い回して本当に同じ結果になるのか」は必ず検証すべきです。""")

    model = Llama(cfg).to(DEVICE).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 8), device=DEVICE)

    torch.manual_seed(7)
    a = model.generate(prompt, 20, temperature=0.0, use_cache=True)
    torch.manual_seed(7)
    b = model.generate(prompt, 20, temperature=0.0, use_cache=False)
    same = torch.equal(a, b)
    print(f"\n  キャッシュ有りと無しの生成結果が一致: {'OK' if same else 'NG'}")
    assert same, "KV キャッシュの実装が間違っています"

    sub("速度比較")
    big = LlamaConfig(dim=384, n_layers=8, n_heads=8, n_kv_heads=2, max_seq_len=512)
    m2 = Llama(big).to(DEVICE).eval()
    p2 = torch.randint(0, big.vocab_size, (1, 16), device=DEVICE)
    print(f"  {'生成トークン数':>14} {'キャッシュ無し':>14} {'キャッシュ有り':>14} {'倍率':>7}")
    for n in [50, 100, 200]:
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        m2.generate(p2, n, temperature=0.0, use_cache=False)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t_no = time.perf_counter() - t0

        t0 = time.perf_counter()
        m2.generate(p2, n, temperature=0.0, use_cache=True)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t_yes = time.perf_counter() - t0
        print(f"  {n:14d} {t_no * 1000:12.0f}ms {t_yes * 1000:12.0f}ms {t_no / t_yes:6.1f}x")
    print("""
  ★ 生成が長くなるほど差が開きます。実運用の LLM サーバは必ずこれを使います。
  ★ 代償がメモリ（検証 3 の表）。速度とメモリのトレードオフを
    GQA で緩和する、という関係になっています。""")

    # ------------------------------------------------------------------
    title("検証 6  モデル全体の健全性チェック")
    # ------------------------------------------------------------------
    model = Llama(cfg).to(DEVICE)
    x = torch.randint(0, cfg.vocab_size, (2, 32), device=DEVICE)
    y = torch.randint(0, cfg.vocab_size, (2, 32), device=DEVICE)
    logits, loss, _ = model(x, y)
    print(f"  出力の形            : {tuple(logits.shape)}")
    print(f"  初期 loss           : {loss.item():.4f}  (理論値 ln(V)={math.log(cfg.vocab_size):.4f})")
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.3

    model.eval()
    a = torch.randint(0, cfg.vocab_size, (1, 24), device=DEVICE)
    b = a.clone()
    b[0, 12:] = torch.randint(0, cfg.vocab_size, (12,), device=DEVICE)
    with torch.no_grad():
        la, _, _ = model(a, targets=a)
        lb, _, _ = model(b, targets=b)
    d = (la[0, :12] - lb[0, :12]).abs().max().item()
    print(f"  因果性（未来の影響）  : 最大差 {d:.2e}  {'OK' if d < 1e-4 else 'NG'}")
    assert d < 1e-4

    sub("パラメータの内訳")
    groups = {}
    for name, p in model.named_parameters():
        key = ("Embedding/出力（共有）" if "tok_embeddings" in name or "output" in name
               else "Attention" if "attention." in name
               else "FFN(SwiGLU)" if "feed_forward" in name
               else "RMSNorm")
        groups[key] = groups.get(key, 0) + p.numel()
    tot = sum(groups.values())
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"    {k:24s} {v:>10,}  ({v / tot * 100:5.1f}%)")
    print(f"    {'合計':24s} {tot:>10,}")

    sub("パラメータ数の見積もり式（自分で設計するとき用）")
    print("""    Decoder-only Transformer の非埋め込みパラメータ数 ≈

        N ≈ 12 * n_layers * dim^2         （MHA + 4倍 FFN の場合の概算）

    内訳: Attention の W_q,W_k,W_v,W_o が 4*dim^2
          FFN が dim*4dim + 4dim*dim = 8*dim^2
    埋め込みは別途 vocab_size * dim。

    例) dim=4096, n_layers=32 -> 12*32*4096^2 ≈ 64.4 億 ≈ LLaMA-2 7B
        （実際は SwiGLU で FFN 部分が 3 層になるので係数が少し変わります）

    ★ 設計の指針:
      ・dim と n_layers の比はおおむね dim ≈ 128 × n_layers が定番です
        （LLaMA-2 7B: 4096 = 128×32層、GPT-3: 12288 = 128×96層）。
        ただし実務では既存モデルの設定をそのままスケールするのが最も安全です。
      ・学習データ量は Chinchilla 則で「パラメータ数の約 20 倍のトークン」。
        215M のモデルなら約 43 億トークンが目安。
        データが足りないなら、モデルを大きくしても loss は下がりません。""")

    title("全検証 完了")


# ======================================================================
def train_demo(steps=1200):
    """ミニコーパスで LLaMA を実際に学習させる。"""
    import _corpus
    from _bpe import ByteLevelBPE

    title("LLaMA アーキテクチャで実際に学習する")
    text = _corpus.make_text(n=20000, seed=0)
    tok = ByteLevelBPE()
    tok.train(text, vocab_size=400)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]

    cfg = LlamaConfig(vocab_size=len(tok), dim=256, n_layers=4,
                      n_heads=8, n_kv_heads=2, max_seq_len=64)
    model = Llama(cfg).to(DEVICE)
    print(f"  デバイス: {DEVICE} / パラメータ: {model.num_params():,}")
    print(f"  語彙: {len(tok)} / 学習トークン: {len(train_ids):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95),
                            weight_decay=0.1)

    def batch(data, bs=32):
        ix = torch.randint(len(data) - cfg.max_seq_len - 1, (bs,))
        x = torch.stack([data[i:i + cfg.max_seq_len] for i in ix]).to(DEVICE)
        y = torch.stack([data[i + 1:i + 1 + cfg.max_seq_len] for i in ix]).to(DEVICE)
        return x, y

    warmup = steps // 20
    print(f"\n  {'step':>6} {'train':>8} {'val':>8} {'経過':>8}")
    t0 = time.perf_counter()
    for step in range(steps):
        lr = 3e-3 * ((step + 1) / warmup if step < warmup else
                     0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup))))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = batch(train_ids)
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vx, vy = batch(val_ids)
                _, vloss, _ = model(vx, vy)
            model.train()
            print(f"  {step:6d} {loss.item():8.4f} {vloss.item():8.4f} "
                  f"{time.perf_counter() - t0:7.1f}s")

    sub("生成（KV キャッシュ使用）")
    # ★ 学習時の文脈長は 64 だが、300 トークン生成する。
    #   RoPE は学習長を超えた位置にも外挿できるため、これが動く
    #   （04_minigpt.py の学習型位置埋め込みでは原理的に不可能だったこと）。
    bos = tok.encode("\n")
    out = model.generate(torch.tensor([bos] * 6, device=DEVICE), 300,
                         temperature=0.9, top_k=30)
    txt = "\n".join(tok.decode(r.tolist()) for r in out)
    for line in txt.split("\n")[1:7]:
        if line.strip():
            print(f"    {line}")
    ok, total, rate = _corpus.score(txt)
    print(f"\n  文法正答率: {ok}/{total} = {rate:.1%}  "
          f"{'[合格]' if rate >= 0.9 else '[要調整]'}")
    print("""
  ★ 04_minigpt.py（GPT-2 構成）とほぼ同じ結果になったはずです。
    小規模では新旧アーキテクチャの差は出ません。差が出るのは
    「長文」「大規模」「推論コスト」の 3 点であり、
    それこそが RoPE・GQA が導入された理由です。""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=1200)
    args = ap.parse_args()
    if args.train:
        train_demo(args.steps)
    else:
        verify()
        print("\n  実際に学習させるには:  python 05_llama.py --train\n")
