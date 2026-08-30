"""
02. Attention を「実際の数字」で最後まで追う
======================================================================
    python 02_attention_step_by_step.py

数式だけ見ても Attention は絶対に腹落ちしません。
4 トークン・4 次元という手で追える大きさで、途中の行列を全部表示します。

    STEP 1  入力を作る（4 トークン、各 4 次元）
    STEP 2  Q, K, V を作る
    STEP 3  QK^T = 類似度スコア行列
    STEP 4  なぜ √d_k で割るのか（分散を実測して確認する）
    STEP 5  softmax で重みにする
    STEP 6  重み × V = 出力
    STEP 7  因果マスク（未来を見せない）
    STEP 8  Multi-Head: 次元を分割する様子を shape で追う
    STEP 9  PyTorch 公式実装と一致することを確認
    STEP 10 「因果性」を実験で証明する

対応ドキュメント: ../02-Transformerアーキテクチャ.md
"""

import math

import torch
import torch.nn.functional as F

from _console import title, sub

torch.manual_seed(42)
torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)

TOKENS = ["猫", "が", "魚", "を"]
T, D = len(TOKENS), 4          # 系列長 4、埋め込み次元 4


def show(name, mat, rows=None, cols=None):
    """行列を行ラベル・列ラベル付きで表示する。"""
    print(f"\n{name}  shape={tuple(mat.shape)}")
    m = mat.detach()
    if cols:
        print(" " * 8 + "".join(f"{c:>9}" for c in cols))
    for i in range(m.shape[0]):
        label = f"{rows[i]:>6}: " if rows else "        "
        print(label + "".join(f"{v:9.3f}" for v in m[i].tolist()))


# ======================================================================
title("STEP 1  入力: 4 トークン × 4 次元のベクトル")
# ======================================================================
X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],    # 猫
    [0.0, 1.0, 0.0, 1.0],    # が
    [1.0, 1.0, 0.0, 0.0],    # 魚
    [0.0, 0.0, 1.0, 1.0],    # を
])
show("X（Embedding 層の出力に相当）", X, rows=TOKENS)
print("""
実際は nn.Embedding(vocab_size, dim) が返すベクトルですが、
ここでは手で追えるように単純な値を置いています。""")


# ======================================================================
title("STEP 2  Q, K, V を作る（同じ X に 3 つの別々の行列を掛ける）")
# ======================================================================
W_q = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])          # 説明用に単位行列
W_k = torch.tensor([[0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0]])          # 次元を入れ替える行列
W_v = torch.eye(4) * 2.0                            # 2 倍にするだけの行列

Q, K, V = X @ W_q, X @ W_k, X @ W_v
show("Q = X @ W_q  （私は何を探しているか）", Q, rows=TOKENS)
show("K = X @ W_k  （私は何として見つけられたいか）", K, rows=TOKENS)
show("V = X @ W_v  （私が渡す中身）", V, rows=TOKENS)
print("""
★ ここが最初の関門: Q・K・V は「別々の入力」ではありません。
   同じ X から、学習可能な 3 つの行列で作った『3 つの見え方』です。
   実務では W_q, W_k, W_v は乱数で初期化され、学習で意味のある値になります。""")


# ======================================================================
title("STEP 3  QK^T = 全トークンペアの類似度")
# ======================================================================
scores_raw = Q @ K.T
show("scores_raw = Q @ K^T", scores_raw, rows=TOKENS, cols=TOKENS)
print(f"""
読み方: scores_raw[i][j] = 「トークン i が トークン j をどれだけ見たいか」
例) scores_raw[0][2] = {scores_raw[0][2]:.3f}
    → 「猫」の Query と「魚」の Key の内積。

なぜ内積か: 内積は 2 つのベクトルの「向きの一致度」です。
    向きが同じ    → 大きい正の値
    直交          → 0
    向きが逆      → 負
つまり「Query が求めている特徴を Key が持っているか」の点数になります。

★ shape に注目: (T, D) @ (D, T) = (T, T)。
   系列長の 2 乗の行列ができます。これが Transformer の計算量が
   O(T^2) で、長文が苦手な理由そのものです。""")


# ======================================================================
title("STEP 4  なぜ √d_k で割るのか（実測して確かめる）")
# ======================================================================
print("""説明: 独立でランダムな d 次元ベクトル同士の内積は、
       次元 d に比例して分散が大きくなります（各項の分散の和になるため）。
       実際に測ってみます。""")
print(f"\n{'次元 d':>8} | {'内積の分散(実測)':>18} | {'√d で割った後':>16}")
print("-" * 52)
for d in [4, 16, 64, 256, 1024]:
    a = torch.randn(20000, d)
    b = torch.randn(20000, d)
    dot = (a * b).sum(-1)
    print(f"{d:8d} | {dot.var().item():18.2f} | {(dot / math.sqrt(d)).var().item():16.3f}")

print("""
→ 分散はほぼ d に一致し、√d で割ると常に 1 に揃います。

なぜ分散が大きいと困るのか: softmax は入力の差を指数で拡大するからです。""")
for scale in [1.0, 5.0, 20.0]:
    z = torch.tensor([1.0, 0.5, 0.2]) * scale
    p = F.softmax(z, dim=-1)
    print(f"  スコア {z.tolist()} → softmax {[round(v,4) for v in p.tolist()]}")
print("""  スケールが大きいと「ほぼ 1 と ほぼ 0」に張り付きます（＝1 箇所しか見ない）。
  softmax が飽和すると勾配がほぼ 0 になり、学習が進まなくなります。
  これを防ぐのが 1/√d_k です。""")

d_k = D
scores = scores_raw / math.sqrt(d_k)
show(f"scores = QK^T / √{d_k}", scores, rows=TOKENS, cols=TOKENS)


# ======================================================================
title("STEP 5  softmax で「重み」にする（各行の合計が 1）")
# ======================================================================
weights = F.softmax(scores, dim=-1)     # ★ dim=-1: 行方向に正規化
show("weights = softmax(scores, dim=-1)", weights, rows=TOKENS, cols=TOKENS)
print(f"各行の合計: {weights.sum(-1).tolist()}  ← 全部 1.0")
print("""
★ dim=-1 の意味: 「トークン i から見た、各 j への配分」を確率にします。
   dim=0 にすると全く違う（意味のない）計算になるので、実装で最も間違えやすい箇所です。
   検算方法: weights.sum(-1) が全部 1 になるかを必ず確認する。""")

print("\n1 行目（「猫」の行）を言葉にすると:")
for j, tok in enumerate(TOKENS):
    bar = "#" * int(weights[0][j].item() * 40)
    print(f"  「猫」→「{tok}」 {weights[0][j].item():.3f}  {bar}")


# ======================================================================
title("STEP 6  出力 = 重み付き平均された V")
# ======================================================================
out = weights @ V
show("out = weights @ V", out, rows=TOKENS)
print(f"""
検算: out の 1 行目は V の各行を weights[0] で混ぜたものです。
  手計算: {sum(weights[0][j].item() * V[j] for j in range(T)).tolist()}
  行列計算: {out[0].tolist()}

★ ここが「文脈を理解する」の正体です。
   「猫」の出力ベクトルの中に、周囲のトークンの情報が
   関連度に応じた比率で混ざり込みました。
   これを 12 層、24 層と重ねることで、複雑な意味が組み上がります。""")


# ======================================================================
title("STEP 7  因果マスク: 未来を見せない")
# ======================================================================
print("""言語モデルは「次のトークンを当てる」訓練をします。
このとき位置 i が位置 i+1 以降を見られると、答えを見ながら答えることになります。
そこで softmax の『前』に、未来の位置のスコアを -inf にします。""")

mask = torch.tril(torch.ones(T, T))          # 下三角が 1
show("mask（1 = 見てよい, 0 = 見てはいけない）", mask, rows=TOKENS, cols=TOKENS)

scores_masked = scores.masked_fill(mask == 0, float("-inf"))
show("マスク適用後の scores", scores_masked, rows=TOKENS, cols=TOKENS)

weights_causal = F.softmax(scores_masked, dim=-1)
show("softmax 後（exp(-inf)=0 なので上三角がちょうど 0）",
     weights_causal, rows=TOKENS, cols=TOKENS)
print("""
★ 順序が超重要: マスクは softmax の『前』。
   softmax の後に 0 を掛けると、行の合計が 1 でなくなり壊れます。
★ -inf を使う理由: softmax は exp を取るので exp(-inf)=0 になり、
   正規化の分母からもきれいに消えます。大きな負の数(-1e9)でも実用上は同じです。""")


# ======================================================================
title("STEP 8  Multi-Head: shape の変形を 1 段ずつ追う")
# ======================================================================
print("""Multi-Head は「Attention を n_heads 個並列に走らせる」ことですが、
実装では『次元を分割する』だけです。ここが読めれば実装で詰まりません。""")

B, T2, C, H = 2, 6, 8, 2                 # batch, seq, dim, heads
head_dim = C // H
x = torch.randn(B, T2, C)
sub("形の変形")
print(f"  入力 x                                 : {tuple(x.shape)}   (B, T, C)")
q = x.view(B, T2, H, head_dim)
print(f"  .view(B, T, H, head_dim)               : {tuple(q.shape)} (B, T, H, hd)  ← C を H 分割")
q = q.transpose(1, 2)
print(f"  .transpose(1, 2)                       : {tuple(q.shape)} (B, H, T, hd)  ← head を batch 側へ")
att = q @ q.transpose(-2, -1)
print(f"  q @ q^T                                : {tuple(att.shape)} (B, H, T, T)   ← head ごとの Attention 行列")
o = att @ q
print(f"  att @ v                                : {tuple(o.shape)} (B, H, T, hd)")
o = o.transpose(1, 2).contiguous().view(B, T2, C)
print(f"  .transpose(1,2).contiguous().view(B,T,C): {tuple(o.shape)}   (B, T, C)   ← 元の形に結合")

print("""
★ .contiguous() が必要な理由:
   transpose はメモリ上のデータを動かさず「見え方」だけ変えます。
   view はメモリが連続していることを要求するため、間に contiguous() が必要です。
   （これを忘れると RuntimeError になる、実装時の定番の落とし穴）

★ 計算量の話: C を H 個に割るので、head 数を増やしても総計算量は同じ。
   「タダで複数の視点を持てる」のが Multi-Head が使われる理由です。""")


# ======================================================================
title("STEP 9  自作 Attention と PyTorch 公式実装を突き合わせる")
# ======================================================================


def my_attention(q, k, v, causal=False):
    """自作版。q,k,v: (B, H, T, hd)"""
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if causal:
        t = q.size(-2)
        m = torch.tril(torch.ones(t, t, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~m, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


q = torch.randn(2, 4, 10, 16)
k = torch.randn(2, 4, 10, 16)
v = torch.randn(2, 4, 10, 16)

for causal in (False, True):
    mine = my_attention(q, k, v, causal=causal)
    official = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    diff = (mine - official).abs().max().item()
    ok = "OK" if diff < 1e-5 else "NG"
    print(f"  causal={str(causal):5s} 最大誤差 = {diff:.3e}  [{ok}]")

print("""
→ 一致。つまり F.scaled_dot_product_attention（Flash Attention）は
   自作版と数学的に同一で、メモリ効率と速度だけが違います。
   実務では公式実装を使いますが、中身は今書いた 4 行と同じです。""")


# ======================================================================
title("STEP 10  因果性を実験で証明する")
# ======================================================================
print("""「未来を見ていない」ことは、こう確かめられます:
   系列の後ろの方だけを書き換えて、前の方の出力が変わらないかを見る。""")

x1 = torch.randn(1, 4, 10, 16)
x2 = x1.clone()
x2[:, :, 7:, :] = torch.randn(1, 4, 3, 16)      # 位置 7 以降だけ差し替え

o1 = F.scaled_dot_product_attention(x1, x1, x1, is_causal=True)
o2 = F.scaled_dot_product_attention(x2, x2, x2, is_causal=True)

front = (o1[:, :, :7] - o2[:, :, :7]).abs().max().item()
back = (o1[:, :, 7:] - o2[:, :, 7:]).abs().max().item()
print(f"\n  位置 0〜6 の出力の差 = {front:.3e}  ← 0 であるべき（未来に影響されない）")
print(f"  位置 7〜9 の出力の差 = {back:.3e}  ← 0 でないはず（自分が変わったので）")
assert front < 1e-6, "因果マスクが効いていません"
print("\n  [OK] 因果マスクは正しく機能しています。")
print("""
★ この検査は自作モデルのデバッグにそのまま使えます。
   ここが 0 にならないモデルは「カンニング」しているので、
   学習時の loss は綺麗に下がるのに生成が全く使い物にならない、という現象が起きます。""")

title("完了")
print("""
Attention でやっていることは、結局この 1 行に尽きます:

    out = softmax( Q @ K^T / √d_k + mask ) @ V

次: python 03_bpe_tokenizer.py
""")
