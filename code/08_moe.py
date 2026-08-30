"""
08. MoE（Mixture of Experts）を作って、数字で確かめる
======================================================================
    python 08_moe.py              # 検証だけ（CPU で 10 秒程度）
    python 08_moe.py --train      # ルータの偏り実験まで含めて実行（CPU で 1〜2 分）

2026 年時点の大きなモデルは、ほぼ全部 MoE（専門家混合）です。
しかし MoE は「新しい仕組み」ではなく、**FFN を N 個に増やして、
トークンごとに数個だけ使う**というだけの改造です。

このスクリプトは、その「だけ」を全部数字で確認します。

    STEP 1  Dense FFN と MoE FFN の「総パラメータ」と「実際に使う量」を数える
    STEP 2  専門家 1 人・top-1 の MoE は Dense FFN と完全に一致することを確認
    STEP 3  ルータが実際にトークンを振り分けている様子を見る
    STEP 4  補助損失なしでルータが偏る（＝専門家の遊び）ことを実測する
    STEP 5  MoE の言語モデルが理論下限まで学習できることを確認する
    STEP 6  KV キャッシュ：MHA / GQA / MQA / MLA の保存量を計算して比べる

対応ドキュメント: ../05.5-MoEと現代アーキテクチャ.md
"""

import argparse
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import _corpus
from _bpe import ByteLevelBPE
from _console import title, sub

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
@dataclass
class MoEConfig:
    dim: int = 128                  # 隠れ次元 d_model
    n_layers: int = 4
    n_heads: int = 4
    vocab_size: int = 400
    block_size: int = 64
    norm_eps: float = 1e-5

    # --- ここから MoE の設定 ---
    n_experts: int = 8              # 専門家（＝FFN）の人数
    n_experts_per_tok: int = 2      # 1 トークンが使う人数（top-k）
    n_shared_experts: int = 0       # 全トークンが必ず通る「共有専門家」の人数
    expert_hidden: Optional[int] = None   # 専門家 1 人ぶんの中間次元
    aux_loss_coef: float = 0.01     # 負荷分散の補助損失の重み（0 で無効）

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


# ======================================================================
# 1. 部品：RMSNorm と SwiGLU FFN（05 章と同じもの）
# ======================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    """専門家 1 人ぶんの FFN。中身は 05 章の FeedForward と完全に同じ。

    MoE で「専門家」と呼ばれているものの正体は、この普通の FFN です。
    特別な仕組みは何も入っていません。
    """

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ======================================================================
# 2. MoE の本体（この 40 行が MoE のすべて）
# ======================================================================
class MoEFeedForward(nn.Module):
    """Dense FFN を「N 人の専門家 ＋ ルータ」に置き換えたもの。

    やっていることは 4 手順だけです。

        ① ルータ（ただの nn.Linear）が、各トークンに専門家の点数を付ける
        ② 点数上位 k 人だけを選ぶ（top-k）
        ③ 選ばれた k 人だけを実行する
        ④ その出力をルータの重み（softmax）で混ぜる

    ★ 大事なのは ③ です。専門家が 8 人いても、1 トークンが通るのは k 人だけ。
      だから「総パラメータは 8 倍、計算量は k 倍」になります。
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.expert_hidden
        if hidden is None:
            # Dense と総計算量を揃えるため、専門家 1 人あたりの中間次元を k で割る
            dense_hidden = cfg.multiple_of_ceil(int(2 * (4 * cfg.dim) / 3))
            hidden = cfg.multiple_of_ceil(max(16, dense_hidden // cfg.n_experts_per_tok), 16)
        self.expert_hidden = hidden

        self.experts = nn.ModuleList(
            [SwiGLU(cfg.dim, hidden) for _ in range(cfg.n_experts)]
        )
        # ルータ（gate）。bias なしの nn.Linear 1 枚だけ
        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)

        # 共有専門家（DeepSeekMoE 系）：全トークンが必ず通る。共通知識の置き場
        self.shared = nn.ModuleList(
            [SwiGLU(cfg.dim, hidden) for _ in range(cfg.n_shared_experts)]
        )

        # 統計（学習後に「誰がどれだけ働いたか」を見るため）
        self.register_buffer("tokens_per_expert", torch.zeros(cfg.n_experts),
                             persistent=False)
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x):
        cfg = self.cfg
        B, T, C = x.shape
        flat = x.reshape(-1, C)                       # (B*T, C) トークンを一列に並べる

        # ① ルータが点数を付ける
        logits = self.router(flat)                    # (N, n_experts)
        probs = F.softmax(logits, dim=-1)

        # ② 上位 k 人を選ぶ。選んだ分だけで再正規化する（Mixtral 方式）
        topv, topi = torch.topk(probs, cfg.n_experts_per_tok, dim=-1)   # (N, k)
        topv = topv / topv.sum(dim=-1, keepdim=True)

        # ③④ 選ばれた専門家だけを実行して、重みで混ぜる
        out = torch.zeros_like(flat)
        counts = torch.zeros(cfg.n_experts, device=x.device)
        for e, expert in enumerate(self.experts):
            # このバッチで専門家 e を選んだトークンの位置
            hit = (topi == e)                         # (N, k) の bool
            if not hit.any():
                continue
            rows = hit.any(dim=-1).nonzero(as_tuple=True)[0]
            weight = (topv * hit).sum(dim=-1)[rows].unsqueeze(-1)
            out[rows] += weight * expert(flat[rows])
            counts[e] = rows.numel()

        for expert in self.shared:                    # 共有専門家は全員が通る
            out = out + expert(flat)

        self.tokens_per_expert = counts.detach()
        self.last_aux_loss = self._aux_loss(probs, topi)
        return out.view(B, T, C)

    def _aux_loss(self, probs, topi):
        """負荷分散の補助損失（GShard / Switch Transformer 方式）。

            aux = n_experts * Σ_e ( f_e × P_e )

            f_e … 専門家 e に実際に振られたトークンの割合
            P_e … ルータが専門家 e に付けた確率の平均

        全員が均等なら f_e = P_e = 1/n となり aux = 1（最小）。
        1 人に集中すると aux は n に近づきます。
        「よく選ばれている専門家の確率をさらに上げる」ことに罰を与える形です。
        """
        n = self.cfg.n_experts
        # f_e: top-k に選ばれた回数の割合
        onehot = F.one_hot(topi, num_classes=n).sum(dim=1).float()      # (N, n)
        f = onehot.mean(dim=0) / self.cfg.n_experts_per_tok
        P = probs.mean(dim=0)
        return n * torch.sum(f * P)


# MoEConfig にヘルパを足す（dataclass の外で定義しても動く）
def _multiple_of_ceil(self, v: int, multiple: int = 32) -> int:
    return multiple * math.ceil(v / multiple)


MoEConfig.multiple_of_ceil = _multiple_of_ceil


# ======================================================================
# 3. MoE を組み込んだ小さな言語モデル（Attention は 02.7 章のものと同じ）
# ======================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: MoEConfig, dense: bool = False):
        super().__init__()
        self.ln1 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.dim, cfg.norm_eps)
        if dense:
            hidden = cfg.multiple_of_ceil(int(2 * (4 * cfg.dim) / 3))
            self.ffn = SwiGLU(cfg.dim, hidden)
        else:
            self.ffn = MoEFeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MoELM(nn.Module):
    def __init__(self, cfg: MoEConfig, dense: bool = False):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg, dense) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.dim, cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))

        loss = aux = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            aux = self.aux_loss()
        return logits, loss, aux

    def aux_loss(self):
        losses = [blk.ffn.last_aux_loss for blk in self.blocks
                  if isinstance(blk.ffn, MoEFeedForward)]
        if not losses:
            return torch.tensor(0.0, device=self.head.weight.device)
        return torch.stack(losses).mean()

    def expert_usage(self):
        """各層の「専門家ごとの担当トークン数」をまとめて返す。"""
        return [blk.ffn.tokens_per_expert.clone() for blk in self.blocks
                if isinstance(blk.ffn, MoEFeedForward)]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.9, top_k=30):
        for _ in range(max_new_tokens):
            cond = idx[:, -self.cfg.block_size:]
            logits, _, _ = self(cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


# ======================================================================
# STEP 1 / 2 / 3 / 6：学習なしで確認できること
# ======================================================================
def verify():
    title("STEP 1  Dense FFN と MoE FFN のパラメータを数える")

    cfg = MoEConfig(dim=128, n_experts=8, n_experts_per_tok=2)
    dense_hidden = cfg.multiple_of_ceil(int(2 * (4 * cfg.dim) / 3))
    dense = SwiGLU(cfg.dim, dense_hidden)
    moe = MoEFeedForward(cfg)

    per_expert = n_params(moe.experts[0])
    active = per_expert * cfg.n_experts_per_tok + n_params(moe.router)
    print(f"""
    dim = {cfg.dim}

    【Dense FFN】     中間次元 {dense_hidden:4d}                総 {n_params(dense):>8,} / 使う {n_params(dense):>8,}
    【MoE  FFN】      専門家 {cfg.n_experts} 人 × 中間次元 {moe.expert_hidden:3d}   総 {n_params(moe):>8,} / 使う {active:>8,}
                      （1 トークンが通るのは top-{cfg.n_experts_per_tok} の {cfg.n_experts_per_tok} 人だけ）

    → 総パラメータは約 {n_params(moe) / n_params(dense):.1f} 倍、
      1 トークンあたりの計算量は約 {active / n_params(dense):.1f} 倍。
      ★ これが MoE のすべてです：「知識の置き場だけ増やし、計算量は増やさない」
""")
    assert active < n_params(moe), "top-k なので、使う量は総量より必ず少ない"

    cfg_sh = MoEConfig(dim=128, n_experts=8, n_experts_per_tok=2, n_shared_experts=1)
    moe_sh = MoEFeedForward(cfg_sh)
    active_sh = per_expert * (cfg_sh.n_experts_per_tok + cfg_sh.n_shared_experts)         + n_params(moe_sh.router)
    print(f"""    【MoE ＋ 共有専門家 1 人】（DeepSeekMoE 系）      総 {n_params(moe_sh):>8,} / 使う {active_sh:>8,}

      共有専門家は「全トークンが必ず通る 1 人」です。ルータに選ばれるかどうかに
      関係なく働くので、**どの入力にも要る共通知識**（助詞の扱いなど）を
      ここに置けます。残りの専門家は分業に専念できる、という分担です。
""")

    # ------------------------------------------------------------------
    title("STEP 2  専門家 1 人・top-1 の MoE は Dense FFN と一致するか")

    one = MoEConfig(dim=64, n_experts=1, n_experts_per_tok=1, expert_hidden=128)
    moe1 = MoEFeedForward(one).eval()
    x = torch.randn(2, 5, 64)
    with torch.no_grad():
        got = moe1(x)
        want = moe1.experts[0](x)      # ただの FFN を直接呼んだ結果
    diff = (got - want).abs().max().item()
    print(f"""
    専門家が 1 人しかいなければ、ルータは softmax で必ず重み 1.0 を出します。
    つまり MoE 層は「ただの FFN」に退化するはずです。

        MoE(x) と FFN(x) の最大差 = {diff:.2e}

    [{'OK' if diff < 1e-6 else 'NG'}] 一致しました。
    → MoE は Attention や損失には一切手を入れていない、FFN だけの改造だと確認できます。
""")
    assert diff < 1e-6

    # ------------------------------------------------------------------
    title("STEP 3  ルータは本当にトークンを振り分けているのか")

    cfg3 = MoEConfig(dim=64, n_experts=4, n_experts_per_tok=2)
    moe3 = MoEFeedForward(cfg3).eval()
    x = torch.randn(1, 6, 64)
    with torch.no_grad():
        logits = moe3.router(x.reshape(-1, 64))
        probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, 2, dim=-1)
        topv = topv / topv.sum(dim=-1, keepdim=True)

    print("\n    （初期化直後なので中身は乱数です。形だけ見てください）\n")
    print("      トークン |  選ばれた専門家  |  混ぜる重み")
    print("      ---------+------------------+---------------------")
    for t in range(6):
        e = topi[t].tolist()
        w = topv[t].tolist()
        print(f"        #{t}     |   E{e[0]}, E{e[1]}         |  {w[0]:.2f}, {w[1]:.2f}"
              f"   （合計 {sum(w):.2f}）")
    print("""
    ★ 重みの合計が必ず 1.00 になっているのがポイントです。
      Attention が「他のトークンを重み付き平均する」のと同じ形で、
      MoE は「専門家の出力を重み付き平均」しています。
      違いは、Attention が全員を混ぜるのに対し MoE は k 人しか混ぜないことだけです。
""")

    # ------------------------------------------------------------------
    title("STEP 6  KV キャッシュの保存量：MHA / GQA / MQA / MLA")

    dim, n_heads, n_layers, seq = 4096, 32, 32, 32768
    head_dim = dim // n_heads
    n_kv_gqa = 8
    lora_rank = 512          # MLA の潜在次元（DeepSeek-V3 は 512）
    b = 2                    # bf16 = 2 バイト

    def cache_gb(n_kv_heads):     # K と V の 2 本ぶん
        return 2 * n_layers * n_kv_heads * head_dim * seq * b / 1024 ** 3

    def cache_gb_mla():           # 潜在ベクトル 1 本だけを貯める
        return n_layers * lora_rank * seq * b / 1024 ** 3

    print(f"""
    設定: dim={dim} / ヘッド {n_heads} / 層 {n_layers} / 文脈 {seq:,} トークン / bf16

      MHA（{n_heads} 組の K,V）        {cache_gb(n_heads):6.2f} GB   … 1 リクエストでこれだけ食う
      GQA（{n_kv_gqa} 組に共有）         {cache_gb(n_kv_gqa):6.2f} GB   … {n_heads // n_kv_gqa} 分の 1
      MQA（1 組に共有）         {cache_gb(1):6.2f} GB   … 極端版（品質は少し落ちる）
      MLA（潜在 {lora_rank} 次元 1 本）   {cache_gb_mla():6.2f} GB   … MHA の約 {cache_gb(n_heads) / cache_gb_mla():.0f} 分の 1

    ★ 「70B のモデルを 140GB のメモリに載せた、で終わり」ではありません。
      長い文脈では KV キャッシュがモデル本体に匹敵する大きさになります。
      GQA / MLA が全部のモデルに入っているのは、この表が理由です。
""")

    # MLA が成り立つ理由（K/V は低ランクで近似できる）を数値で示す
    sub("なぜ潜在ベクトル 1 本に圧縮できるのか（低ランク性の確認）")
    torch.manual_seed(0)
    keep = 64

    def kept_energy(m):
        sv = torch.linalg.svdvals(m)
        return (sv[:keep].pow(2).sum() / sv.pow(2).sum()).item()

    # ① 「見た目 1024 次元だが、実質 64 次元ぶんの情報しかない」行列
    core = torch.randn(2048, 64) @ torch.randn(64, 1024)
    lowrank = core + 0.02 * core.std() * torch.randn(2048, 1024)   # 少しノイズを足す
    # ② 本当に 1024 次元ぶん情報が詰まっている行列（比較用）
    fullrank = torch.randn(2048, 1024)

    print(f"""
    どちらも 2048 × 1024 の行列ですが、上位 {keep} 成分だけを残すと——

      ① 実質 {keep} 次元ぶんの情報しかない行列  … 元の情報の {kept_energy(lowrank) * 100:5.1f}% が残る
      ② 本当に 1024 次元ぶん情報がある行列      … 元の情報の {kept_energy(fullrank) * 100:5.1f}% しか残らない

    MLA がやっているのは ① への賭けです。**K/V をいったん細い潜在ベクトルに
    落として保存し、使うときに各ヘッドへ広げ直す**。
    GQA が「ヘッドを共有する」で減らしたのに対し、MLA は「次元そのものを圧縮する」で減らします。

    ⚠️ ②が示すとおり、低ランク性は自動的に成り立つ性質ではありません。
      MLA は「K/V は低ランクに潰しても情報がほとんど失われない」という
      **経験的な観察に賭けた設計**であり、そこは学習で確かめるしかない部分です。
""")
    assert kept_energy(lowrank) > 0.9 > kept_energy(fullrank)


# ======================================================================
# STEP 4 / 5：実際に学習させて確かめる
# ======================================================================
def build_data(vocab_size=400, n_sentences=20000):
    text = _corpus.make_text(n=n_sentences)
    tok = ByteLevelBPE()
    tok.train(text, vocab_size=vocab_size)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(ids))
    # 理論下限（02.7 章と同じ計算）
    min_loss = n_sentences * math.log(180) / len(ids)
    return tok, ids[:n], ids[n:], min_loss


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def eval_loss(model, data, cfg, iters=20):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y = get_batch(data, cfg.block_size, 32)
        _, loss, _ = model(x, y)
        tot += loss.item()
    model.train()
    return tot / iters


def train_one(cfg, train_ids, val_ids, steps, lr=3e-3, label=""):
    torch.manual_seed(0)
    model = MoELM(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    for step in range(steps):
        x, y = get_batch(train_ids, cfg.block_size, 32)
        _, loss, aux = model(x, y)
        total = loss + cfg.aux_loss_coef * aux
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    val = eval_loss(model, val_ids, cfg)
    return model, val


def usage_report(model, tag):
    """専門家ごとの担当割合を可視化する。

    見るべき指標は 2 つです。
      ① 最も忙しい専門家の担当割合（偏りの大きさ）
      ② 「ほぼ遊んでいる」専門家の数（理想の 1/4 未満しか担当していない人数）
    ② が多いと、増やしたパラメータがそのまま無駄になります。
    """
    usage = model.expert_usage()
    n = model.cfg.n_experts
    ideal = 100 / n
    dead_th = ideal / 4
    print(f"\n    【{tag}】各層で、専門家ごとに何 % のトークンを担当したか")
    print("      層 | " + " ".join(f"E{i:<2}" for i in range(n)) + " |  最大   遊び")
    print("      ---+" + "-" * (4 * n) + "-+-------------")
    maxima, dead = [], 0
    for li, u in enumerate(usage):
        share = (u / u.sum() * 100)
        maxima.append(share.max().item())
        d = int((share < dead_th).sum().item())
        dead += d
        cells = " ".join(f"{v:3.0f}" for v in share.tolist())
        print(f"       {li} | {cells} | {share.max():4.0f}%  {d} 人")
    mean_max = sum(maxima) / len(maxima)
    print(f"\n      理想（完全に均等）= 全員 {ideal:.0f}%"
          f"  ／  最大の偏り（層平均）= {mean_max:.0f}%"
          f"  ／  遊んでいる専門家 = 全 {n * len(usage)} 人中 {dead} 人")
    return mean_max, dead


def train_demo(steps=800):
    title("学習データの準備")
    tok, train_ids, val_ids, min_loss = build_data()
    print(f"""
    02.7 章と同じ人工コーパス（文法規則が既知なので機械採点できる）を使います。
      語彙          : {len(tok)}
      学習トークン   : {len(train_ids):,}
      理論下限 loss  : {min_loss:.4f}
""")

    base = dict(dim=128, n_layers=4, n_heads=4, vocab_size=len(tok),
                block_size=64, n_experts=8, n_experts_per_tok=2)

    # ------------------------------------------------------------------
    title("STEP 4  補助損失を切ると、ルータは偏るのか")
    print("""
    MoE の最大の実務課題は「専門家の遊び」です。
    ルータは学習可能なので、放っておくと **少数の専門家だけを選び続ける**
    方向に進みがちです（選ばれた人だけが上手くなり、さらに選ばれる、の循環）。
    そうなると、増やしたパラメータが働かないまま残ります。

    本当に起きるのか、補助損失の重みだけを変えて 2 回学習して比べます。
""")
    cfg_off = MoEConfig(**base, aux_loss_coef=0.0)
    m_off, val_off = train_one(cfg_off, train_ids, val_ids, steps, label="aux=0")
    max_off, dead_off = usage_report(m_off, "補助損失なし（aux_loss_coef = 0）")

    cfg_on = MoEConfig(**base, aux_loss_coef=0.01)
    m_on, val_on = train_one(cfg_on, train_ids, val_ids, steps, label="aux=0.01")
    max_on, dead_on = usage_report(m_on, "補助損失あり（aux_loss_coef = 0.01）")

    n_slots = cfg_on.n_experts * cfg_on.n_layers
    print(f"""
    ------------------------------------------------------------------
                    | val loss | 偏り(層平均) | 遊んでいる専門家
      補助損失なし   |  {val_off:.4f}  |    {max_off:3.0f}%      |  {dead_off} / {n_slots} 人
      補助損失あり   |  {val_on:.4f}  |    {max_on:3.0f}%      |  {dead_on} / {n_slots} 人
    ------------------------------------------------------------------

    ★ 注目すべきは val loss ではなく右の 2 列です。
      **loss はほとんど同じなのに、働いている専門家の数が全く違います。**
      補助損失なしの側は、増やしたパラメータの多くが遊んだまま終わっています。

    ★ つまり loss だけ見ていても偏りには気づけません。
      MoE を学習させるときは **必ず専門家の担当割合をログに出す** ——
      これが 02.7 章の「初期 loss ≈ ln(V)」と並ぶ、MoE 版のバグ検出法です。

    📝 このコーパスは規則が 180 通りしかない極小のタスクなので、
      dense でも十分に解けてしまい「MoE の性能上の利点」は出ません。
      ここで確認できるのは **仕組みと失敗モード**だけです。それが目的です。
""")

    # ------------------------------------------------------------------
    title("STEP 5  MoE でも理論下限まで学習できるか")
    model = m_on
    print(f"""
      最終 val loss = {val_on:.4f}
      理論下限      = {min_loss:.4f}
      差            = {val_on - min_loss:+.4f}
""")
    bos = tok.encode("\n")
    start = torch.tensor([bos] * 8, dtype=torch.long, device=DEVICE)
    out = model.generate(start, 400, temperature=0.9, top_k=30)
    text = "\n".join(tok.decode(r.tolist()) for r in out)
    ok, total, rate = _corpus.score(text)
    print(f"      生成した文: {total} 文中 {ok} 文が文法的に正しい  →  正答率 {rate:.1%}")
    print(f"      判定: {'合格' if rate >= 0.9 else '要調整（--steps を増やす）'}")
    print("""
    ★ MoE にしても、02.7 章・05 章とまったく同じ場所（理論下限）に着地します。
      MoE は「賢くする」技術ではなく、**同じ計算量でパラメータを増やす**技術です。
      効果が出るのは、データが膨大で dense では容量が足りないときだけです。
""")


# ======================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="STEP 4/5（学習あり）も実行する")
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()

    verify()
    if args.train:
        train_demo(args.steps)
    else:
        print("\n  ルータの偏りの実験まで見るには:  python 08_moe.py --train\n")
