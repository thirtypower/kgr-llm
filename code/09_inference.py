"""
09. 推論を速くする：KV キャッシュ・バッチ・量子化を実測する
======================================================================
    python 09_inference.py            # 速度の実測だけ（CPU で 1〜2 分）
    python 09_inference.py --quant    # 量子化の品質劣化まで測る（+ 約 1 分）

学習の話は 05／06 章で終わりました。実務で本当に金と時間を食うのは、
**学習より推論（inference）** です。ここでは「なぜ速くなるのか」を
全部自分の手元の数字で確認します。

    STEP 1  KV キャッシュ：結果が一致することを確認 → 何倍速くなるか実測
    STEP 2  キャッシュなしのコストが「文が長くなるほど」増えることを実測
    STEP 3  バッチを増やすとスループットがどう伸びるかを実測
    STEP 4  重みを N bit に落とすと loss がどれだけ悪化するかを実測（--quant）
    STEP 5  実物サイズでの KV キャッシュ量と、投機的デコードの期待値を計算

02.7 章で作ったモデル（04_minigpt.py）をそのまま部品として使います。

対応ドキュメント: ../07.5-推論を速くする.md
"""

import argparse
import importlib
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import _corpus
from _bpe import ByteLevelBPE
from _console import title, sub

mg = importlib.import_module("04_minigpt")     # 02.7 章のモデルをそのまま借りる
GPTConfig, MiniGPT = mg.GPTConfig, mg.MiniGPT

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# 生成の2つの実装（違いは「過去を計算し直すかどうか」だけ）
# ======================================================================
@torch.no_grad()
def generate_naive(model, idx, n_new):
    """キャッシュなし。毎回「文の全体」を forward し直す（00 章の素朴な方式）。"""
    model.eval()
    for _ in range(n_new):
        cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(cond)                  # ★ 毎回 T トークン全部を計算
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


@torch.no_grad()
def generate_cached(model, idx, n_new):
    """KV キャッシュあり。過去の K/V を貯めて、新しい 1 トークンだけ計算する。

    ★ 02.7 章の検査5 と同じ実装です。ポイントは 2 つだけ。
       ① 各層の (K, V) を caches に貯めて、次の呼び出しに渡す
       ② 位置埋め込みには「いま何トークン目か」を渡す（★ここを間違えると壊れる）
    """
    model.eval()
    caches = [(None, None)] * model.cfg.n_layer

    def step(tokens, start_pos):
        pos = torch.arange(start_pos, start_pos + tokens.size(1), device=tokens.device)
        h = model.tok_emb(tokens) + model.pos_emb(pos)
        for i, blk in enumerate(model.blocks):
            h, caches[i] = blk(h, caches[i])
        return model.head(model.ln_f(h))

    logits = step(idx, 0)                        # ① プロンプト部分を一括処理（prefill）
    for _ in range(n_new):
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        logits = step(nxt, idx.size(1) - 1)      # ② 以降は 1 トークンずつ（decode）
    return idx


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


def make_model(block_size=2048, n_layer=8, n_embd=768, vocab=400):
    cfg = GPTConfig(vocab_size=vocab, block_size=block_size, n_layer=n_layer,
                    n_head=12, n_embd=n_embd, dropout=0.0)
    return MiniGPT(cfg).to(DEVICE).eval(), cfg


# ======================================================================
# STEP 1  KV キャッシュ
# ======================================================================
def step1():
    title("STEP 1  KV キャッシュ：まず「同じ結果になる」ことを確認する")

    model, cfg = make_model()
    # ★ すでに 1024 トークン書いた状態から続きを書く、という設定にします。
    #   （プロンプトが短いと、どちらの方式でも計算量が小さく差が出ません）
    T0, n_new = 1024, 32
    prompt = torch.randint(0, cfg.vocab_size, (1, T0), device=DEVICE)

    a = generate_naive(model, prompt.clone(), n_new)
    b = generate_cached(model, prompt.clone(), n_new)
    same = torch.equal(a, b)
    print(f"""
    最適化を入れたら、まず「最適化前と結果が一致するか」を確かめます。
    貪欲法（argmax）で生成すれば結果は決定的になるので、完全一致するはずです。

        キャッシュなしの出力 == キャッシュありの出力 : {same}

    [{'OK' if same else 'NG'}] 一致しました。
    → これが 02.7 章の検査5 です。**速くなったかを測る前に、壊れていないかを測る。**
""")
    assert same, "KV キャッシュの実装が間違っています（位置のずれが最有力）"

    sub("① 何トークンぶんの計算をしたか（数え上げ：ハードウェアに依存しない）")
    # キャッシュなしは毎ステップで「それまでの全トークン」を計算し直す
    naive_tokens = sum(T0 + i for i in range(n_new))
    cached_tokens = T0 + n_new            # prefill で T0 回 + 1 トークンずつ n_new 回
    print(f"""
      条件: すでに {T0} トークンある状態から、さらに {n_new} トークン生成

      キャッシュなし : {naive_tokens:>7,} トークンぶんを forward
                      （{T0}+{T0 + 1}+…+{T0 + n_new - 1} と毎回作り直すため）
      キャッシュあり : {cached_tokens:>7,} トークンぶんを forward
                      （最初の {T0} は 1 回だけ。以降は 1 トークンずつ）

                      → 無駄な計算は 約 {naive_tokens / cached_tokens:.0f} 倍

    ★ この比は**モデルの大きさにも GPU にも依存しない、純粋な数え上げ**です。
      本物の LLM（数千トークンの入力に数千トークン生成）では
      **数百〜数千倍**になります。だから KV キャッシュは「あると速い」ではなく
      「ないと実用にならない」機能なのです。
""")

    sub("② 実測時間（GPT-2 small 相当・約57Mパラメータのモデルで計測）")
    _, t_naive = timed(generate_naive, model, prompt.clone(), n_new)
    _, t_cache = timed(generate_cached, model, prompt.clone(), n_new)
    print(f"""
      キャッシュなし : {t_naive:6.2f} 秒  （{n_new / t_naive:6.1f} トークン/秒）
      キャッシュあり : {t_cache:6.2f} 秒  （{n_new / t_cache:6.1f} トークン/秒）
                       → 実測 約 {t_naive / t_cache:.1f} 倍速

    📝 **①の「{naive_tokens / cached_tokens:.0f} 倍」より実測の倍率が小さいのは正常です。**
      1 ステップには行列積以外の固定コスト（PyTorch の関数呼び出しなど）が数 ms あり、
      キャッシュあり側はその固定コストが下限になってしまうからです。
      **モデルを大きくするほど、実測は①の理論値に近づきます**
      （逆に極小モデルにすると、差はほとんど消えます。自分で `make_model` の
       `n_embd` を 128 に下げて試してみてください）。

    🔑 **これ自体が実務で重要な教訓です**：
      **小さいモデルで測った速度ベンチマークは、大きいモデルの挙動を予測しません。**
      「何が律速なのか」（固定オーバーヘッド／計算／メモリ帯域）が
      規模によって入れ替わるからです。

    ★ そして KV キャッシュが効く理由自体は単純です。
      **過去のトークンの K と V は毎回まったく同じ値**なので、
      計算し直す必要が無かった——それだけです（因果マスクのおかげで、
      過去は未来の影響を受けないから）。
""")
    return model, cfg


# ======================================================================
# STEP 2  「長くなるほど遅くなる」を実測する
# ======================================================================
def step2(model, cfg):
    title("STEP 2  文が長くなると、1トークンの値段はどう変わるか")

    print("""
    キャッシュなしの生成は、毎回「これまでの全トークン」を forward します。
    つまり 1000 トークン目を出すには 1000 トークンぶんの計算が必要です。

    キャッシュありなら、毎回 1 トークンぶんしか計算しません。
    ——では、1 トークンの値段は**一定になる**のでしょうか？

    すでに書いた長さを変えて「あと 1 トークンだけ」生成し、時間を測ります。
""")
    print("      すでに書いた長さ |  キャッシュなし  |  キャッシュあり  |  差")
    print("      -----------------+------------------+------------------+------")
    for T in (32, 128, 512, 1024, 1900):
        ctx = torch.randint(0, cfg.vocab_size, (1, T), device=DEVICE)

        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            generate_naive(model, ctx.clone(), 1)
        t_n = (time.perf_counter() - t0) / reps * 1000

        # キャッシュありは「prefill が終わった状態から 1 トークン」を測る
        caches = [(None, None)] * cfg.n_layer

        def step(tokens, start_pos):
            pos = torch.arange(start_pos, start_pos + tokens.size(1), device=DEVICE)
            h = model.tok_emb(tokens) + model.pos_emb(pos)
            for i, blk in enumerate(model.blocks):
                h, caches[i] = blk(h, caches[i])
            return model.head(model.ln_f(h))

        with torch.no_grad():
            step(ctx, 0)                                   # prefill（測定対象外）
            one = ctx[:, -1:].clone()
            t0 = time.perf_counter()
            for _ in range(reps):
                saved = [(k, v) for k, v in caches]        # 毎回同じ条件で測る
                step(one, T)
                caches[:] = saved
            t_c = (time.perf_counter() - t0) / reps * 1000

        print(f"      {T:>13} tok | {t_n:9.2f} ms    | {t_c:9.2f} ms    |"
              f" {t_n / t_c:4.1f}x")

    print("""
    ★ **答えは「一定にはならない」です。** 両方とも長さと一緒に増えています。
      ただし増え方が違います（差の列を見てください。長くなるほど開いていく）。

      ここが理解の分かれ目なので、値段を 2 つに分けて考えます。

        (A) Embedding・FFN・各種の射影  … トークン 1 個ごとに固定の計算
        (B) Attention                  … 「過去の全トークン」を相手にする計算

      キャッシュなし : (A) も (B) も、毎回すべてのトークンについて計算し直す
      キャッシュあり : (A) は新しい 1 トークンぶんだけ。**(B) だけは過去全部が必要**

    🔑 **KV キャッシュが消せるのは (A) だけです。(B) は原理的に消せません。**
      新しいトークンは、過去の全トークンとの関連度を計算しなければならないからです
      （02章の Attention の定義そのもの）。しかも計算し直さないだけで、
      **過去の K と V をメモリから読み出す**必要は残ります。

    ★ だから **長い文脈は、キャッシュを入れても高いままです。**
      そして「(B) をどう安くするか」が、05.5 章で見た
      **GQA / MLA（読む量を減らす）** と
      **sliding window / 線形 Attention（相手の数を減らす）** の動機になります。
      この表の右の列の伸びが、あれらの技術が生まれた理由です。
""")


# ======================================================================
# STEP 3  バッチとスループット
# ======================================================================
def step3(model, cfg):
    title("STEP 3  バッチを増やすと、なぜ「ほぼタダ」でスループットが伸びるのか")

    print("""
    生成の 1 ステップは「1 トークンぶんの行列積」です。これは行列としては
    極端に細長く（1 × dim）ので、GPU の演算器はほとんど遊んでいます。
    **重みをメモリから読み込む時間**の方が支配的だからです（STEP 5 ②）。

    ということは——**同時に何本もの文を生成しても、1 ステップの時間は
    バッチ数ほどには増えない**はずです（重みは 1 回読めば全員で使い回せる）。実測します。
""")
    n_new = 32
    # 最初の呼び出しはスレッド起動などで遅いので、捨てる（計測の常識）
    generate_cached(model, torch.randint(0, cfg.vocab_size, (8, 8), device=DEVICE), 8)

    print("      バッチ |  1ステップの時間  |  スループット   |  1本あたりの値段")
    print("      -------+-------------------+-----------------+------------------")
    base = None
    for B in (1, 4, 16, 64):
        prompt = torch.randint(0, cfg.vocab_size, (B, 8), device=DEVICE)
        generate_cached(model, prompt, 4)                  # ウォームアップ
        _, t = timed(generate_cached, model, prompt, n_new)
        per_step = t / n_new * 1000
        thr = B * n_new / t
        if base is None:
            base = per_step
        print(f"      {B:>5}  | {per_step:11.2f} ms   | {thr:9.0f} tok/s | "
              f"{per_step / base:5.2f} 倍")

    print("""
    ★ バッチを 64 倍にしても、1 ステップの時間は 64 倍にはなりません。
      つまり **同時利用者が増えるほど、1 人あたりのコストは下がる**。
      これが LLM API が個人で動かすより安い最大の理由です。
      （実務ではこれを最大限に使うために、到着した要求を次々と同じバッチに
        差し込む **continuous batching** という仕組みを使います → 07.5 章）

    📝 ここは CPU での実測なので、伸びの内訳は GPU とは違います
      （CPU では「1 ステップあたりの固定オーバーヘッド」の寄与が大きい）。
      ただし **「バッチを N 倍にしても時間は N 倍にならない」という結論は同じ**で、
      GPU ではその理由が「decode がメモリ帯域律速だから」になります。

    ⚠️ 代わりに増えるのが **KV キャッシュのメモリ**です（人数分だけ必要）。
      「バッチをどこまで大きくできるか」＝「KV キャッシュを何本置けるか」であり、
      GQA や MLA（→ 05.5 章）が効いてくるのはまさにここです。
""")


# ======================================================================
# STEP 4  量子化の品質劣化を測る
# ======================================================================
def fake_quantize_(model, bits, skip_embedding=True):
    """重みを N bit に丸めて、また float に戻す（per-channel absmax 方式）。

    実物の量子化ライブラリ（GPTQ / AWQ など）はもっと賢いことをしますが、
    「精度を落とすと何が起きるか」を見るにはこの最も素朴な方式で十分です。
    """
    qmax = 2 ** (bits - 1) - 1
    for name, p in model.named_parameters():
        if p.dim() < 2:                       # LayerNorm などの 1 次元は触らない
            continue
        if skip_embedding and "emb" in name:  # Embedding は感度が高いので除外（実務も同じ）
            continue
        scale = p.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        p.data = torch.round(p.detach() / scale).clamp(-qmax, qmax) * scale


def step4(steps=400):
    title("STEP 4  重みの精度を落とすと、賢さはどれだけ失われるか")

    print("""
    推論を安くする最短の道は「重みの精度を落とす」ことです。
    bf16（2バイト）→ 4bit（0.5バイト）なら、モデルサイズは 4 分の 1。
    ただし当然、無料ではありません。**どれくらい払うのか**を測ります。

    02.7 章のモデルを軽く学習させてから、重みだけを N bit に丸めて
    **val loss と、生成文の文法正答率**を測り直します。
    loss は「少し悪くなった」しか教えてくれませんが、正答率は
    **実際に使い物になるかどうか**を教えてくれます。
""")
    torch.manual_seed(1234)          # 何回実行しても同じ数字が出るように固定
    text = _corpus.make_text(n=20000, seed=0)
    tok = ByteLevelBPE()
    tok.train(text, vocab_size=400)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]
    min_loss = 20000 * math.log(180) / len(ids)

    cfg = GPTConfig(vocab_size=len(tok), block_size=64, n_layer=4, n_head=4,
                    n_embd=128, dropout=0.0)
    model = MiniGPT(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95),
                            weight_decay=0.1)
    model.train()
    for step in range(steps):
        x, y = mg.get_batch(train_ids, cfg.block_size, 32, DEVICE)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    @torch.no_grad()
    def val_loss(m):
        m.eval()
        tot = 0.0
        for _ in range(30):
            x, y = mg.get_batch(val_ids, cfg.block_size, 32, DEVICE)
            _, l = m(x, y)
            tot += l.item()
        return tot / 30

    @torch.no_grad()
    def grammar_rate(m):
        """生成させて、文法規則に合っている文の割合を測る（02.7 章と同じ採点）。"""
        m.eval()
        torch.manual_seed(7)                       # 生成の乱数も固定
        bos = tok.encode("\n")
        start = torch.tensor([bos] * 8, dtype=torch.long, device=DEVICE)
        out = m.generate(start, 200, temperature=0.9, top_k=30)
        txt = "\n".join(tok.decode(r.tolist()) for r in out)
        _, _, rate = _corpus.score(txt)
        return rate

    base, base_rate = val_loss(model), grammar_rate(model)
    sd = {k: v.clone() for k, v in model.state_dict().items()}

    print("\n      精度      |  val loss  |  文法正答率  |  モデルの大きさ")
    print("      ----------+------------+--------------+-----------------")
    print(f"      fp32(元)  |   {base:.4f}   |   {base_rate:6.1%}     |   1.00 倍")
    for bits in (8, 6, 4, 3, 2):
        model.load_state_dict(sd)
        fake_quantize_(model, bits)
        print(f"      {bits:>2} bit    |   {val_loss(model):.4f}   |"
              f"   {grammar_rate(model):6.1%}     |   {bits / 32:.2f} 倍")

    print(f"""
      （理論下限 = {min_loss:.4f} / 学習は 400 step に短縮しているので、
        02.7 章の 98% より正答率の出発点が低くなっています）

      📝 正答率は生成のサンプリングを含むので **±数 % は誤差**です
        （3bit が fp32 より高く出ているのはそれ。意味はありません）。
        意味があるのは「2bit だけが半分に落ちた」という段差の方です。

    ★ 読み方：**ある bit 数まではほぼ無料で、そこを割ると急に落ちる。**
      「じわじわ悪くなる」のではなく **崖がある** のがポイントです。
      そして **loss の悪化はごくわずかでも、正答率は大きく落ちることがあります** ——
      **loss だけ見て「まだ大丈夫」と判断してはいけない**、ということです。

    ⚠️ **この表を「4bit でも 3bit でも無料」と読まないでください。**
      ここでの課題は 180 通りの文型を覚えるだけで、モデルには余裕がありすぎます。
      本物の LLM で実際に報告されている傾向はこうです。
        - int8 / FP8  … 品質劣化はほぼ無視できる（本番で標準的に使われる）
        - int4        … 多くの用途で許容範囲。ただし**数学・コード・長い推論**で劣化が出やすい
        - 3bit 以下   … 素朴な方式では壊れる。実用には GPTQ / AWQ のように
                        **校正データで誤差を補正しながら量子化する**手法が必要

    🔑 **持ち帰るべきは数字ではなく手順です**：
      量子化したら**必ず自分のタスクで測り直す**。
      「4bit にしても大丈夫」は、タスクごとに確かめる以外に知る方法がありません。
""")


# ======================================================================
# STEP 5  実物サイズでの計算
# ======================================================================
def step5():
    title("STEP 5  実物サイズで計算してみる")

    sub("① KV キャッシュはどれだけメモリを食うか")
    # LLaMA-2 7B 相当
    n_layers, n_heads, head_dim = 32, 32, 128
    print(f"""
    LLaMA-2 7B 相当（層 {n_layers} / ヘッド {n_heads} / head_dim {head_dim} / bf16）

      KV キャッシュ = 2 × 層 × KVヘッド × head_dim × トークン数 × 2 バイト
""")
    print("      文脈長    |  MHA(32組)  |  GQA(8組)  |  MQA(1組)")
    print("      ----------+-------------+------------+-----------")
    for seq in (2048, 8192, 32768, 131072):
        def gb(kv):
            return 2 * n_layers * kv * head_dim * seq * 2 / 1024 ** 3
        print(f"      {seq:>8,} | {gb(32):8.2f} GB | {gb(8):7.2f} GB | {gb(1):6.2f} GB")
    print("""
    ★ モデル本体は bf16 で約 14GB です。
      **128K トークンの MHA では、1 人ぶんのキャッシュがモデル本体の 4 倍以上**。
      「長い文脈は高い」の正体はこれです（そして GQA が必須になる理由）。
""")

    sub("② prefill と decode は、まったく性質の違う処理")
    print("""
                 | 何をするか                    | 律速するもの      | 増やすと得なもの
      -----------+-------------------------------+-------------------+------------------
      prefill    | 入力プロンプト全体を1回で処理  | **計算能力**      | 演算器を使い切る
      （読む）    | 行列積が「大きい行列 × 行列」  | (compute bound)   | → 最初の1文字までの待ち時間
      -----------+-------------------------------+-------------------+------------------
      decode     | 1トークンずつ生成             | **メモリ帯域**    | バッチサイズ
      （書く）    | 行列積が「1行 × 行列」で細長い | (memory bound)    | → スループット

    ★ 「最初の1文字が出るまでの待ち時間（TTFT）」と
      「そのあと1秒に何文字出るか（TPS）」は、**別のボトルネックで決まります**。
      遅いと感じたとき、どちらが遅いのかを分けて測るのが第一歩です。
""")

    sub("③ 投機的デコード（speculative decoding）の期待値")
    print("""
    decode がメモリ帯域律速なら、**1 回の forward で 2 トークン以上確定できれば得**です。
    そこで小さい下書きモデルに k トークン先まで書かせ、
    本命モデルが「その k トークンを1回の forward でまとめて検証」します。
    合っていた分だけ採用、間違ったところから書き直し。

      期待して得られるトークン数（受理率 α、下書き k トークン）
          E = (1 − α^(k+1)) / (1 − α)
""")
    print("      受理率 α |  k=2  |  k=4  |  k=8   ← 1回の検証で確定するトークン数")
    print("      ---------+-------+-------+-------")
    for a in (0.5, 0.7, 0.8, 0.9):
        vals = [(1 - a ** (k + 1)) / (1 - a) for k in (2, 4, 8)]
        print(f"        {a:.1f}    | {vals[0]:5.2f} | {vals[1]:5.2f} | {vals[2]:5.2f}")
    print("""
    ★ 受理率 0.8 なら、1 回の検証で平均 3〜4 トークン確定できます。
      **出力は元のモデルと数学的に同一**（検証で弾くので）なのに速くなる——
      だから「品質を落とさない高速化」として本番で広く使われています。
      逆に受理率が低い（下書きが弱い・出力が予測しにくい）と、
      検証のコストだけ増えて**遅くなる**ことに注意してください。
""")


# ======================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", action="store_true", help="STEP 4（量子化）も実行する")
    ap.add_argument("--steps", type=int, default=400, help="STEP 4 の学習ステップ数")
    args = ap.parse_args()

    model, cfg = step1()
    step2(model, cfg)
    step3(model, cfg)
    if args.quant:
        step4(args.steps)
    else:
        print("\n  量子化の劣化まで測るには:  python 09_inference.py --quant\n")
    step5()
