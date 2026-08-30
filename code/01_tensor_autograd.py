"""
01. 「学習する」とは何をしているのか
======================================================================
このファイルを実行すると、LLM の学習で起きていることを
最小の例で最初から最後まで見ることができます。

    python 01_tensor_autograd.py

扱う内容:
    STEP 1  テンソルとは何か
    STEP 2  勾配（微分）を手計算する
    STEP 3  同じものを autograd に計算させる
    STEP 4  勾配降下法で「学習」させる
    STEP 5  分類問題の損失 = 交差エントロピー（手計算 vs PyTorch）
    STEP 6  次トークン予測は「語彙数クラスの分類」だと確認する
    STEP 7  パープレキシティ（損失の直感的な読み方）
    STEP 8  Optimizer（SGD と Adam）の違いを目で見る

対応ドキュメント: ../00.5-深層学習の基礎.md
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from _console import title   # 標準出力を UTF-8 にする副作用つき

torch.manual_seed(0)


# ======================================================================
# STEP 1  テンソルとは何か
# ======================================================================
title("STEP 1  テンソルとは「多次元の数値の箱」でしかない")

scalar = torch.tensor(3.0)                       # 0次元 = ただの数
vector = torch.tensor([1.0, 2.0, 3.0])           # 1次元 = ベクトル
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2次元 = 行列
tensor3 = torch.zeros(2, 3, 4)                   # 3次元

for name, t in [("scalar", scalar), ("vector", vector),
                ("matrix", matrix), ("tensor3", tensor3)]:
    print(f"{name:8s} shape={tuple(t.shape)}  次元数={t.dim()}  要素数={t.numel()}")

print("""
LLM では次のような形が出てきます:
    (batch, seq_len)            ... トークン ID の列
    (batch, seq_len, dim)       ... 各トークンのベクトル表現（隠れ状態）
    (batch, heads, seq, seq)    ... Attention の重み行列
shape を追えるようになることが、実装できるようになることの 8 割です。
""")


# ======================================================================
# STEP 2  勾配を手計算する
# ======================================================================
title("STEP 2  勾配（＝どちらに動かせば損失が減るか）を手で計算する")

print("""
「重み w が 1 つだけのモデル」を考えます。

    予測:  y_pred = w * x
    正解:  y_true
    損失:  L = (y_pred - y_true)^2          ← 二乗誤差

L を w で微分すると（合成関数の微分）:

    dL/dw = 2 * (w*x - y_true) * x

この dL/dw が「勾配」です。
勾配は "w を +1 動かしたら L がどれだけ増えるか" を表します。
だから L を減らしたいなら、勾配と『逆』方向へ動かせばよい。
""")

x, y_true, w = 2.0, 10.0, 1.0
y_pred = w * x
loss = (y_pred - y_true) ** 2
grad_manual = 2 * (w * x - y_true) * x

print(f"x={x}, y_true={y_true}, w={w}")
print(f"  y_pred = w*x            = {y_pred}")
print(f"  loss   = (y_pred-y)^2   = {loss}")
print(f"  dL/dw  = 2*(w*x-y)*x    = {grad_manual}   ← 負なので w を増やせば損失が減る")


# ======================================================================
# STEP 3  autograd に同じことをやらせる
# ======================================================================
title("STEP 3  autograd（自動微分）: PyTorch は微分を自動でやる")

w_t = torch.tensor([1.0], requires_grad=True)   # ★ requires_grad=True が肝
x_t = torch.tensor([2.0])
y_t = torch.tensor([10.0])

y_pred_t = w_t * x_t
loss_t = (y_pred_t - y_t) ** 2
loss_t.backward()                                # ★ ここで全ての勾配が計算される

print(f"手計算   : {grad_manual}")
print(f"autograd : {w_t.grad.item()}")
assert abs(w_t.grad.item() - grad_manual) < 1e-6
print("→ 一致。以降、微分は PyTorch に任せてよい。")

print("""
仕組み（計算グラフ）:
    w ──×x──▶ y_pred ──(-y)──▶ diff ──^2──▶ loss
    forward で「どんな演算をしたか」が記録され、
    backward で逆向きに連鎖律（chain rule）を適用して各変数の勾配を求める。
    これが「誤差逆伝播（backpropagation）」の正体です。
""")


# ======================================================================
# STEP 4  勾配降下法 = 学習ループ
# ======================================================================
title("STEP 4  勾配降下法: 勾配の逆方向に少しずつ動かす")

print("""
    w ← w - lr * dL/dw
lr（learning rate / 学習率）は 1 歩の大きさ。
これを何万回も繰り返すのが「学習」です。LLM でも全く同じです。
""")

w_t = torch.tensor([1.0], requires_grad=True)
lr = 0.1
print(f"{'step':>4} {'w':>10} {'loss':>10} {'grad':>10}")
for step in range(8):
    loss_t = (w_t * x_t - y_t) ** 2
    if w_t.grad is not None:
        w_t.grad.zero_()          # ★ 勾配は累積するので毎回ゼロにする
    loss_t.backward()
    print(f"{step:4d} {w_t.item():10.4f} {loss_t.item():10.4f} {w_t.grad.item():10.4f}")
    with torch.no_grad():         # 更新自体は微分の対象外
        w_t -= lr * w_t.grad

print(f"\n最終 w = {w_t.item():.4f}  （正解は y/x = {y_true/x:.1f}）")


# ======================================================================
# STEP 5  分類の損失 = 交差エントロピー
# ======================================================================
title("STEP 5  交差エントロピー: 「正解の確率をどれだけ高く言えたか」")

print("""
LLM の出力は「次のトークンの確率分布」です。つまり語彙数クラスの分類問題。
分類の損失には交差エントロピー（cross entropy）を使います。

    1) モデルの生スコア            : logits          （実数、範囲自由）
    2) softmax で確率にする        : p_i = exp(z_i) / Σ exp(z_j)
    3) 正解クラスの確率の -log     : L = -log(p_正解)

なぜ -log か:
    p=1.0 なら L=0        （完璧 → 罰なし）
    p=0.5 なら L=0.693
    p=0.1 なら L=2.303
    p→0   なら L→∞        （正解を「ありえない」と言ったら大罰）
""")

logits = torch.tensor([[2.0, 1.0, 0.1]])   # 3 クラス分のスコア
target = torch.tensor([0])                 # 正解はクラス 0

probs = F.softmax(logits, dim=-1)
manual_loss = -torch.log(probs[0, target[0]])
torch_loss = F.cross_entropy(logits, target)

print(f"logits         : {logits.tolist()[0]}")
print(f"softmax 後の確率: {[round(v, 4) for v in probs.tolist()[0]]}  （合計={probs.sum():.4f}）")
print(f"手計算 -log(p0) : {manual_loss.item():.6f}")
print(f"F.cross_entropy: {torch_loss.item():.6f}")
assert torch.allclose(manual_loss, torch_loss, atol=1e-6)
print("→ 一致。F.cross_entropy は「softmax + -log」を 1 つにまとめたもの。")
print("  （数値安定性のため内部で log_softmax を使う。自分で softmax してから")
print("    渡すと二重適用になるので注意！ logits をそのまま渡すのが正しい）")


# ======================================================================
# STEP 6  次トークン予測は「語彙数クラスの分類」
# ======================================================================
title("STEP 6  LLM の損失計算をミニチュアで再現する")

vocab_size, batch, seq_len = 1000, 2, 5
targets = torch.randint(0, vocab_size, (batch, seq_len))  # 1つ後ろのトークン

# 初期化直後のモデルは「どのトークンも同じくらいありえる」と言う状態。
# 実際の初期化（std=0.02 程度の小さな乱数）を再現すると logits はほぼ 0 になる。
logits = torch.randn(batch, seq_len, vocab_size) * 0.02

print(f"logits  shape = {tuple(logits.shape)}   (batch, seq_len, vocab_size)")
print(f"targets shape = {tuple(targets.shape)}      (batch, seq_len)")

# cross_entropy は (N, C) と (N,) を要求するので、位置を全部つぶす
loss = F.cross_entropy(
    logits.view(-1, vocab_size),   # (batch*seq_len, vocab_size)
    targets.view(-1),              # (batch*seq_len,)
)
print(f"\n初期化直後の loss           = {loss.item():.4f}")
print(f"理論値 ln(vocab_size)      = {math.log(vocab_size):.4f}")

# 参考: 初期化が「大きすぎる」とどうなるか
bad = torch.randn(batch, seq_len, vocab_size) * 3.0
bad_loss = F.cross_entropy(bad.view(-1, vocab_size), targets.view(-1))
print(f"初期化が大きすぎる場合の loss = {bad_loss.item():.4f}  ← 明らかに ln(V) より大きい")

print("""
★ 実務で最重要のチェック:
   学習開始直後の loss が ln(語彙数) 付近なら、初期化と損失計算は正しい。
   例) 語彙 32000 → ln(32000) ≈ 10.37 から下がり始めるのが正常。
   ・ln(V) より大幅に大きい → 初期化のスケールが大きすぎる
   ・いきなり異常に小さい   → ラベルがリークしている（入力に正解が混入）
   この 1 行の確認だけで、実装バグの半分は開始 10 秒で見つかります。
""")

# ラベルマスク（SFT で使う）
targets_masked = targets.clone()
targets_masked[:, :3] = -100          # 前半 3 トークンを損失計算から除外
loss_masked = F.cross_entropy(logits.view(-1, vocab_size),
                              targets_masked.view(-1),
                              ignore_index=-100)
print(f"前半をマスクした loss = {loss_masked.item():.4f}")
print("→ ignore_index=-100 の位置は損失に寄与しない。これが SFT の『損失マスク』。")


# ======================================================================
# STEP 7  パープレキシティ
# ======================================================================
title("STEP 7  パープレキシティ = exp(loss) = 「何択で迷っているか」")

for L in [10.37, 5.0, 3.0, 2.0, 1.0]:
    print(f"  loss={L:5.2f}  →  perplexity={math.exp(L):10.1f}  "
          f"（実質 {math.exp(L):.0f} 択で迷っている感覚）")
print("""
GPT-2 級で ppl 20 前後、現代の大規模 LLM で 5〜10 程度。
loss が 0.1 下がることの意味が、ppl に直すと実感しやすくなります。
""")


# ======================================================================
# STEP 8  Optimizer の違い
# ======================================================================
title("STEP 8  SGD と AdamW: 同じ問題を解かせて比べる")

print("""
SGD  : w ← w - lr * g                       （素朴。パラメータごとの事情を無視）
Adam : 勾配の移動平均 m と、勾配二乗の移動平均 v を持ち
       w ← w - lr * m / (√v + eps)
       → 勾配が小さいパラメータは大きく、暴れるパラメータは小さく動かす
AdamW: Adam の weight decay を正しく分離したもの。LLM の事実上の標準。
""")


print("""実験: わざと「スケールがバラバラな特徴量」を与えます。
（1列目は 0.001 倍、4列目は 1000 倍。実際の深層モデルでも層ごとに
  勾配の大きさは桁違いにバラつくので、これは現実的な状況です）
""")

# 列ごとにスケールが 10^6 倍も違うデータ
SCALE = torch.tensor([0.001, 1.0, 10.0, 1000.0])
CHECKPOINTS = (0, 49, 99, 199)


def run(opt_name: str, lr: float, steps: int = 200):
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    data_x = torch.randn(256, 4) * SCALE
    data_y = (data_x[:, 3] > 0).long() + (data_x[:, 0] > 0).long()   # 解ける規則
    if opt_name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                weight_decay=0.01)
    history = []
    for i in range(steps):
        loss = F.cross_entropy(model(data_x), data_y)
        opt.zero_grad(set_to_none=True)   # ① 勾配リセット
        loss.backward()                   # ② 勾配計算
        opt.step()                        # ③ パラメータ更新
        if i in CHECKPOINTS:
            history.append(loss.item())
    return history


# SGD は学習率をどこに合わせても破綻することを見せる（これが本質）
settings = [("SGD",   1e-6, "小さい lr: 大きい列に合わせた"),
            ("SGD",   1e-3, "大きい lr: 小さい列に合わせた"),
            ("AdamW", 1e-2, "")]
print(f"{'optimizer':>10} {'lr':>7} | {'step 0':>9} {'step 50':>9} {'step 100':>9} {'step 200':>9}")
print("-" * 62)
for name, lr, note in settings:
    h = run(name, lr)
    row = " ".join(f"{v:9.4f}" for v in h)
    print(f"{name:>10} {lr:7.0e} | {row}   {note}")
print("""
読み方:
  SGD lr=1e-6 : ほぼ動かない。スケール 1000 の列には適切な歩幅だが、
                スケール 0.001 の列は 100 万倍遅くしか学習されない。
  SGD lr=1e-3 : 下がったと思ったら跳ね上がる（発散と回復を繰り返す）。
                小さい列には適切でも、大きい列にとっては歩幅が大きすぎる。
  AdamW       : 何も調整していないのに素直に収束する。

SGD は「全パラメータに同じ歩幅」なので、スケールがバラバラだと
どこに lr を合わせても、どこかの列が壊れます。
Adam は各パラメータを自分の勾配の大きさ（√v）で割るので、この差を自動で吸収します。
LLM は層数も多く勾配のスケールが層ごとに全然違うため、AdamW がほぼ必須です。""")
print("""
★ 学習ループの 3 点セットは必ずこの順:
      opt.zero_grad()  →  loss.backward()  →  opt.step()
   zero_grad を忘れると勾配が累積し、学習が意味不明に壊れます
   （逆に、意図的に累積させるのが「勾配累積 gradient accumulation」）。
""")

title("完了：ここまでが『学習』の全てです")
print("""
LLM の学習も、規模が違うだけでやっていることはこの 8 ステップと同一です。
    ・入力から予測を出す（forward）
    ・正解との差を交差エントロピーで測る（loss）
    ・autograd で全パラメータの勾配を得る（backward）
    ・AdamW で少し動かす（step）
    ・これを数十万回繰り返す

次: python 02_attention_step_by_step.py
""")
