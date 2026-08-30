---
title: "05. 自分で LLM を作る（LLaMA2 の実装）"
---

> **この章のゴール**：LLaMA2 のアーキテクチャを PyTorch でゼロから書き、
> トークナイザを訓練し、**215M パラメータの小さな LLM を実際に学習させる**。
>
> 📎 **対応コード**：[`code/05_llama.py`](https://github.com/thirtypower/kgr-llm/blob/main/code/05_llama.py)（5.1〜5.2 の検証）／
> [`code/06_pretrain.py`](https://github.com/thirtypower/kgr-llm/blob/main/code/06_pretrain.py)（5.3 の事前学習パイプライン完全版）
> `04_minigpt.py` で作った **GPT-2 相当のモデルを、LLaMA2 に「現代化」していく**構成です。
> 変更点は5つだけで、**それぞれ「本当に効いているか」を数値で確認**します。
>
> ```bash
> cd code
> python 05_llama.py            # 全部の検証を実行（CPU/GPU どちらでも数十秒）
> python 05_llama.py --train    # ミニコーパスで実際に学習まで行う
> ```

ここから実践パートです。ここまでの理論が、コードとしてどう現れるかを見ていきます。

> ⚠️ **この章がいきなり難しく感じたら、先に [02.7章](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/027-mini-gpt) を読んでください。**
> 素の GPT-2 を CPU だけで数分で学習させる章です。
> **「動くモデルを1回作った」状態でこの章に来ると、難易度が体感で半分になります。**
> この章がやっているのは、そこで作った GPT-2 の部品を5個入れ替える作業だからです。
>
> ```
>   02.7章の GPT-2          この章の LLaMA2
>   ─────────────────────────────────────────
>   LayerNorm         →     RMSNorm      （高速化）
>   学習型 位置埋め込み  →     RoPE         （相対位置・長文対応）
>   MHA               →     GQA          （KVキャッシュの保存量を削減）
>   ReLU の FFN       →     SwiGLU       （表現力）
>   バイアス項         →     全部なくす     （不要と分かった）
> ```

---

## 5.0 この章で作るもの

| 項目 | 値 |
|------|-----|
| アーキテクチャ | LLaMA2（Decoder-only Transformer） |
| パラメータ数 | **215M（2.15億）** |
| 次元 `dim` | 1024 |
| 層数 `n_layers` | 18 |
| 語彙数 `vocab_size` | 6144 |
| 最大系列長 | 512 |
| 事前学習データ | 日本語 Wikipedia など（5.3.1 で用意） |
| SFT データ | 日本語の指示-応答データ（5.3.1 で用意） |

> 💡 **2026年の実物とのギャップについて**：この章で作るのは
> **dense（MoE でない）モデル**です。いま公開されている大型モデルはほぼ全部 MoE で、
> KV キャッシュもさらに圧縮（MLA）されています。
> **その差分だけを集めた補講が [05.5章](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/055-moe-modern-arch)** です。
> ただし骨格（Pre-Norm / RMSNorm / RoPE / SwiGLU）はこの章のままなので、
> **先にこの章を通してください。**

> 💡 学習に使うデータは 5.3.1 で用意します。日本語 Wikipedia や
> [llm-jp-corpus](https://gitlab.llm-jp.nii.ac.jp/) など、好きなコーパスに差し替えられます
> （5.3.1 に実際のダウンロードスクリプトがあります）。

### 本文コードと `code/` スクリプトの対応

この章には「2つのバージョン」のコードが登場します。混乱しないように先に整理しておきます。

| | 本文のコード | [`code/05_llama.py`](https://github.com/thirtypower/kgr-llm/blob/main/code/05_llama.py) | [`code/06_pretrain.py`](https://github.com/thirtypower/kgr-llm/blob/main/code/06_pretrain.py) |
|---|---|---|---|
| 位置づけ | 写経用のリファレンス実装（**215M 設計**） | CPU で数分で動く**検証・学習デモ** | 5.3 節の学習パイプラインの**完全に動く版** |
| 規模 | dim=768〜1024 / vocab=6144 | dim=256 / vocab=400 | `--demo`＝ミニ構成 / `--full`＝215M 構成 |
| 追加実装 | — | KV キャッシュ、`verify()`（数値検証） | データDL・トークナイザ訓練・checkpoint 保存 |
| 用途 | 写経して構造を理解する | 各技術の効果を数値で確認する | 実際に事前学習を1周する |

クラス名も少し違います（役割は同じ）：

| 本文 | code/05_llama.py | 役割 |
|------|-----------------|------|
| `ModelConfig` | `LlamaConfig` | ハイパーパラメータ |
| `Transformer` | `Llama` | モデル全体 |
| `DecoderLayer` | `TransformerBlock` | Decoder 1層 |
| `MLP` | `FeedForward` | SwiGLU FFN |

### 必要リソースの目安（誇張なしの現実的な数字）

| やること | 必要な環境 | 目安 |
|---------|-----------|------|
| この章を読む・写経する | なんでも可 | — |
| `code/05_llama.py` / `06_pretrain.py --demo` | **CPU で十分** | 数十秒〜数分 |
| **215M の事前学習**（`06_pretrain.py --full`） | GPU **VRAM 12GB〜**（RTX 3060 12GB で可） | 10B トークンを回すと**数週間規模**。**1〜2B トークンに絞れば数日**、まず 0.1B（約1億）トークンで1晩回して感触を掴むのが現実的 |
| 215M の SFT | 同上（VRAM 12GB〜） | 数時間（データ数万件の場合） |

> 💡 GPU が無い場合も、この章の内容は `--demo` で全て体験できます。
> 「215M を本当に回す」のは Colab（無料 T4 でも可、時間はかかる）という手もあります。

---

## 5.1 LLaMA2 を実装する

### 5.1.1 ハイパーパラメータの定義

まずモデルの設定をまとめるクラスを作ります。

```python
from dataclasses import dataclass
from typing import Optional

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
```

`n_heads`（16）> `n_kv_heads`（8）となっている点に注目してください。
これが **GQA（Grouped-Query Attention）** の設定です。

---

### 5.1.2 RMSNorm

第2章の LayerNorm を簡略化したものです。**LLaMA 以降の標準**。

**LayerNorm との違い：平均を引く処理を省略する。**

```
LayerNorm:  y = γ · (x - mean) / std + β        ← 平均と分散の両方を計算
RMSNorm:    y = γ · x / RMS(x)                   ← 二乗平均平方根だけ

  RMS(x) = √( (1/n) Σ xᵢ² )
```

**電卓で1回やってみます。** `x = [2, −1, 4, 3]`（4次元）のとき：

```mermaid
flowchart LR
    X["<b>入力 x</b><br/>[2, −1, 4, 3]"]

    X --> L1["<b>LayerNorm</b><br/>① 平均を出す　mean = 2.0<br/>② 平均を引く　[0, −3, 2, 1]<br/>③ 標準偏差で割る　std = 1.871"]
    L1 --> L2["<b>[0.000, −1.604, 1.069, 0.535]</b><br/><i>平均 0 / 標準偏差 1</i>"]

    X --> R1["<b>RMSNorm</b><br/>① 二乗平均の平方根を出す<br/>　 RMS = √((4+1+16+9)/4) = 2.739<br/>② 割る　（引き算は<b>しない</b>）"]
    R1 --> R2["<b>[0.730, −0.365, 1.461, 1.095]</b><br/><i>平均 0.730（0 にならない）<br/>ベクトルの長さは √次元 に揃う</i>"]

    classDef io fill:#f6f6f6,stroke:#999
    classDef ln fill:#e8f0fe,stroke:#4a7ec4,stroke-width:2px
    classDef rn fill:#e9f7ef,stroke:#3d9970,stroke-width:2px
    class X io
    class L1,L2 ln
    class R1,R2 rn
```

**2つの出力を見比べると、違いは「平行移動したか」だけ**です。
どちらも「大きさを揃える」仕事はできていて、RMSNorm は**平均を求める1パスを省いている**だけです。

| | LayerNorm | RMSNorm |
|---|---|---|
| 引き算（中心化） | する | **しない** |
| 出力の平均<br/><i>（γ・β を掛ける前）</i> | 必ず 0 | 0 にならない |
| 出力の大きさ<br/><i>（同上）</i> | 標準偏差が 1 | **ベクトルの長さが √次元**（上の例では 2.0） |
| 学習パラメータ | γ と β（2本） | **γ だけ**（1本） |
| 次元ぶんの走査 | 平均で1回 ＋ 分散で1回 | **1回** |

**なぜこれで良いのか**：研究の結果、LayerNorm の効果の大半は
「平均を0にすること」ではなく「**スケールを揃えること**」だと分かりました。
平均の計算を省くことで **7〜64% 高速化**しつつ、性能は変わりません。

> 🔑 **正規化の目的は「次の層に渡す数字の大きさを一定に保つこと」**です
> （→ [02.5章 2.2.3](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/025-transformer-block)）。
> 大きさが揃っていれば、平均がどこにあるかは後段の重みが吸収できます。
> だから「引き算をやめても壊れなかった」わけです。

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # 学習可能なスケール γ

    def _norm(self, x):
        # x.pow(2).mean(-1) = 二乗平均、rsqrt = 1/√x
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)
```

> 💡 `eps` は LLaMA 公式実装でも **1e-5 と 1e-6 の間で揺れがあります**
> （LLaMA-2 は 1e-5、初代 LLaMA は 1e-6）。どちらでも学習結果はほぼ変わりません。
> この教材では `ModelConfig.norm_eps` と `code/05_llama.py` に合わせて **1e-5** で統一します。

---

### 5.1.3 RoPE（回転位置埋め込み）

第2章の sin/cos 位置エンコーディングの後継です。**LLaMA / Qwen / 現代 LLM の標準**。

#### 考え方

従来は「位置ベクトルを**足す**」でしたが、RoPE は
**「Q と K のベクトルを、位置に応じた角度だけ**回転させる**」** という方式です。

まず「回転」を絵にします。**次元を2個だけ取り出して**、
1トークン進むごとに θ = 30° ずつ回していったところです。

```
   時計の文字盤と同じです（θ = 30° の場合、1周ちょうど12トークン）

                          位置3
                位置4               位置2
         位置5                             位置1

      位置6                ●                  位置0

         位置7                            位置11
                位置8              位置10
                          位置9

   位置0            位置1(+30°)       位置2(+60°)       位置3(+90°)
     →                  ↗                 ↗                 ↑
  [1.00, 0.00]     [0.87, 0.50]     [0.50, 0.87]     [0.00, 1.00]
     └──────── 長さはいつも 1（回転は長さを変えない）────────┘
```

**足し算と違って、値が大きくなっていきません。** 位置 10000 でも、
ベクトルは同じ円の上のどこかにいるだけです。

> ❓ **文字盤なら、位置12 は位置0 と同じ向きに戻ってしまいませんか？**
> はい、**1つの回転速度だけでは位置が一意になりません**（12個ごとに一巡します）。
> だから RoPE は **24組（head_dim=48 の場合）のペアを全部違う速さで回します**。
> 「秒針・分針・時針が全部同じ位置に戻る」ことがめったに無いのと同じ理屈で、
> 24本の針が同時に一致することは実質起きません。
> 各ペアの速さは、この節の後半（「ペアごとの回る速さ」の表）で実際に数えます。

#### 何を 64 個ぶん回すのか（実装の全体像）

RoPE は head_dim の数字を**2個ずつのペア**に分け、
**ペアごとに違う速さで回します**。ここが 02.5章 の sin/cos と同じ「秒針と時針」の発想です。
5.1.1 の既定値（`dim=768` / `n_heads=16`）なら head_dim は 48 なので、**24 組**になります
（5.0 の 215M 構成は `dim=1024` なので head_dim 64 ＝ **32 組**。以下は 48 で通します）。

```mermaid
flowchart TB
    Q["<b>Q（または K）1ヘッドぶん</b><br/>head_dim = 48 個の数字"]
    P["<b>2個ずつ 24 組のペアに分ける</b><br/>(x₀,x₁) (x₂,x₃) … (x₄₆,x₄₇)<br/><i>＝ 24 枚の2次元平面</i>"]
    R0["<b>ペア0</b> を 位置×57.3° 回す<br/><i>速い＝近距離の区別担当</i>"]
    R1["<b>ペア12</b> を 位置×0.573° 回す"]
    R2["<b>ペア23</b> を 位置×0.0084° 回す<br/><i>遅い＝遠距離の区別担当</i>"]
    OUT["<b>回した 48 個を元の並びに戻す</b><br/>→ あとは普通に QKᵀ を計算するだけ<br/><i>Attention 本体は1行も変わらない</i>"]

    Q --> P
    P --> R0
    P --> R1
    P --> R2
    R0 --> OUT
    R1 --> OUT
    R2 --> OUT

    classDef io fill:#e8f0fe,stroke:#4a7ec4,stroke-width:2px
    classDef rot fill:#fff4e6,stroke:#e8912d
    classDef out fill:#e9f7ef,stroke:#3d9970,stroke-width:2px
    class Q,P io
    class R0,R1,R2 rot
    class OUT out
```

> 📝 **V は回しません。**回すのは Q と K だけです。
> 位置を効かせたいのは「どこを見るか（スコア）」であって、
> 「何を持ってくるか（中身）」ではないからです。

#### なぜ優れているのか

2つのベクトルの内積を取ると、**回転角の差だけが残ります**。

```mermaid
flowchart LR
    A["<b>Q</b> を位置 m ぶん回す<br/>角度 ＋m·θ"]
    B["<b>K</b> を位置 n ぶん回す<br/>角度 ＋n·θ"]
    C["<b>内積を取る</b><br/>＝ |Q||K|·cos( 元の角度差 ＋ (m−n)·θ )"]
    D["<b>絶対位置 m と n は消え、<br/>差 (m−n) だけが残る</b><br/><i>＝ 相対位置！</i>"]

    A --> C
    B --> C
    C --> D

    classDef io fill:#e8f0fe,stroke:#4a7ec4,stroke-width:2px
    classDef key fill:#e9f7ef,stroke:#3d9970,stroke-width:3px
    class A,B,C io
    class D key
```

つまり、**絶対位置で回転させているのに、内積の結果は相対位置で決まる**という性質があります。

#### 電卓だけで確かめる（2次元・1ペアだけ）

「回転角の差だけが残る」を、[02章 2.1.4](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/02-transformer-attention) と同じやり方で実際に確かめます。
**2次元のベクトル1組で足ります。**

**準備**：2次元ベクトルを角度 θ だけ回す計算はこれだけです（座標を混ぜるだけ）。

```
  [x, y] を θ 回転  →  [ x·cosθ − y·sinθ ,  x·sinθ + y·cosθ ]
```

**設定**：Q も K も、回転する前は同じ向き `[1, 0]` だとします。
1回転ぶんの角度を θ = 30° として、**位置3の Q** と **位置1の K** の内積を計算します。

```
  位置3の Q ＝ [1,0] を 3θ = 90° 回した   → [cos90°, sin90°] = [0.00, 1.00]
  位置1の K ＝ [1,0] を 1θ = 30° 回した   → [cos30°, sin30°] = [0.87, 0.50]

  内積 = 0.00×0.87 + 1.00×0.50 = 0.50
```

**次に、両方の位置を2つずつ後ろにずらして、まったく同じ計算をします。**

```
  位置5の Q ＝ [1,0] を 5θ = 150° 回した  → [−0.87,  0.50]
  位置3の K ＝ [1,0] を 3θ =  90° 回した  → [ 0.00,  1.00]

  内積 = (−0.87)×0.00 + 0.50×1.00 = 0.50    ← ★ さっきと同じ値！
```

> 🔑 **(3,1) でも (5,3) でも内積は 0.50。**
> どちらも**位置の差が 2**だからです（`cos(3θ−1θ) = cos(5θ−3θ) = cos(2θ) = cos60° = 0.50`）。
>
> **絶対位置（3 と 5）は内積から完全に消え、差（2）だけが効いている。**
> これが「RoPE は相対位置を自然に扱える」の正体です。
> 回転は**長さを変えない**ので、`[1,0]` を何度回しても内積が壊れない（＝どこまで位置が進んでも
> 値が爆発しない）のもポイントです。

**02章の sin/cos と何が違うのか**：

| | 02章の位置エンコーディング | RoPE |
|---|---|---|
| やり方 | 位置ベクトルを Embedding に **足す** | Q と K を位置の分だけ **回す** |
| 効くのは | 絶対位置（何番目か） | **相対位置（何個離れているか）** |
| 学習長を超えたとき | 足す値が未知の領域に入る | **回転角の付け方を調整するだけで伸ばせる**（下の ⚠️ 参照） |

| 利点 | 説明 |
|------|------|
| 相対位置を自然に扱える | 「3つ前の単語」といった関係を学習しやすい |
| 追加パラメータ不要 | 学習するパラメータが増えない |
| 文脈長を後から伸ばせる | 位置の表し方が「回転角」なので、**重みを作り直さずに**伸ばす細工ができる |

> ⚠️ **よくある誤解：「RoPE は長文にそのまま外挿できる」——これは正しくありません。**
>
> **素の RoPE は、学習した長さを超えると性能が急激に落ちます。**
> 回転角は `位置 × 周波数` なので、学習中に一度も見ていない**大きな回転角**が現れ、
> モデルはその角度での振る舞いを学んでいないからです。
>
> 長文を扱える実際のモデルは、例外なく**RoPE を伸ばす追加の細工**を入れています。
>
> | 手法 | 中身 |
> |---|---|
> | **PI（位置補間）** | 位置 `m` を `m/s` に薄めて、既知の角度の範囲に押し込む。近い位置の区別がぼやける |
> | **NTK-aware / YaRN** ★現在の標準 | **速い波（近距離担当）は触らず、遅い波（遠距離担当）だけ薄める** |
>
> 正しい言い方はこうです——
> **「RoPE は位置を回転で表しているので、重みを作り直さずに文脈長を伸ばす細工ができる。
> ただし細工なしでは伸びない」**。
> 詳しくは [05.5章 5.5.3](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/055-moe-modern-arch) で扱います。

#### ★ ペアごとの「回る速さ」を実際に数える（YaRN の伏線）

上の警告と 05.5章 を理解する鍵は、**ペアごとに回転速度が桁違いに違う**という事実です。
`head_dim = 48`、`theta = 10000` で計算した実数値です。

| ペア番号 | 1トークン進むと回る角度 | 1周（360°）に必要なトークン数 | 役割 |
|---|---|---|---|
| ペア0 | **57.30°** | 約 **6** トークン | 隣か2つ隣か、を区別する |
| ペア1 | 39.04° | 約 9 | ↑ |
| ペア4 | 12.34° | 約 29 | 句レベル |
| ペア8 | 2.66° | 約 135 | 文レベル |
| ペア12 | 0.573° | 約 628 | 段落レベル |
| ペア18 | 0.0573° | 約 6,283 | 文書レベル |
| ペア23 | **0.0084°** | 約 **42,807** トークン | ほぼ動かない |

```
   ペア0  ●→◐→○→◑→●→◐   6トークンで1周（速い＝細かい距離）
   ペア12 ●→●→●→●→●→●    628トークンでやっと1周（遅い＝大まかな距離）
```

> 🔑 **ここから、2つの重要な事実がまとめて出てきます。**
>
> 1. **なぜ「学習した長さを超えると壊れる」のか**
>    この章のモデルは `max_seq_len = 512` です。その 512 トークンの間に、
>    ペア18 は **29.3° しか進みません**（1周の 8%）。ペア23 は **4.3°** だけです。
>    つまり**遅いペアは、学習中に「ごく狭い角度の範囲」しか経験していません**。
>    文脈を 8倍に伸ばせば、そこは**訓練中に一度も見たことがない角度**になります。
>    → 壊れるのは**遅い波の方**、というのが次の項の出発点です。
> 2. **なぜ YaRN は「遅い波だけ薄める」のか**
>    速い波（ペア0付近）は 6 トークンで1周してしまうので、学習中にあらゆる角度を経験済みです。
>    ここを薄めると**近い距離の区別が壊れる**だけで、得るものがありません。
>    薄めるべきは、未知の角度に入る**遅い波の方**です。→ [05.5章 5.5.3](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/055-moe-modern-arch)

#### 実装

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """回転角を事前計算する（複素数表現）"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)                       # 位置 0, 1, 2, ...
    freqs = torch.outer(t, freqs).float()       # (位置, 次元/2)
    # 大きさ1・角度 freqs の複素数を作る（＝回転を表す）
    return torch.polar(torch.ones_like(freqs), freqs)

def reshape_for_broadcast(freqs_cis, x):
    """freqs_cis (T, hd/2) を x (B, T, H, hd/2) に掛けられる形 (1, T, 1, hd/2) にする"""
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), \
        f"freqs_cis {tuple(freqs_cis.shape)} が x {tuple(x.shape)} と合いません"
    shape = [d if i in (1, x.ndim - 1) else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)                     # (1, T, 1, hd/2)

def apply_rotary_emb(xq, xk, freqs_cis):
    """Q, K に回転を適用する。xq, xk: (バッチ, 系列長, ヘッド数, head_dim)"""
    # 実数ベクトルを2次元ずつペアにして複素数とみなす
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # ★ 形を (1, T, 1, hd/2) に整えてから掛ける。ここを省くと必ず落ちる（下の警告参照）
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # 複素数の掛け算 ＝ 回転
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

> 💡 複素数が出てきますが、怖がる必要はありません。
> 「2次元ベクトルの回転は、複素数の掛け算で書ける」という数学の便利技を使っているだけです。
> 上の手計算では `[x·cosθ − y·sinθ, …]` と座標を混ぜましたが、
> **複素数の掛け算1回がその計算とまったく同じ**なので、コードが1行で済むわけです。

#### コードで起きている「形の変化」だけ追う

このコードで迷うのは数学ではなく **shape** です。やっていることは
**「48個の数字を、2個ずつ24組のペアにして、各ペアを回して、また48個に戻す」**だけです
（`head_dim = dim / n_heads = 768 / 16 = 48`）。

```
  ① 入ってくる Q           (B, T, H, 48)          ← head_dim = 48
       ↓ .reshape(..., -1, 2)     48個を「2個ずつ」に割る
  ②                        (B, T, H, 24, 2)       ← 24組のペア
       ↓ torch.view_as_complex(...)  末尾の2個を「1つの複素数」とみなす
  ③                        (B, T, H, 24)          ← 複素数24個（実質は同じ数値）
       ↓ × freqs_cis  (1, T, 1, 24)   複素数の掛け算 ＝ 24組を同時に回転
  ④                        (B, T, H, 24)          ← 回転後
       ↓ torch.view_as_real(...)     複素数を実数2個に戻す
  ⑤                        (B, T, H, 24, 2)
       ↓ .flatten(3)                 3番目の軸から後ろを平らに潰す
  ⑥ 出ていく Q             (B, T, H, 48)          ← ★ 入ってきた形に戻った
```

> 🔑 **`view_as_complex` / `view_as_real` は「見方を変える」だけで、数値も個数も変えません。**
> `flatten(3)` の `3` は「3番目の軸（0から数えて）より後ろをくっつける」という指定で、
> `(…, 24, 2)` を `(…, 48)` に戻しています。
> **入口と出口の形が同じ**（`(B, T, H, 48)`）なので、RoPE を挟んでも後ろの処理は何も変わりません。
>
> 📝 head_dim が偶数でなければならないのは、この「2個ずつペアにする」ができないからです。
> LLaMA 系の設定で head_dim が 64・128 のような偶数になっているのは、この都合です。
>
> 💡 なぜ「2個ずつペア」なのか——**回転は2次元でしか定義できない**からです。
> 64次元のベクトルを、32枚の2次元平面に分けて、それぞれ別の速さで回している
> （速さの違いが `freqs_cis` の中身で、02.5章の sin/cos の「速い波・遅い波」と同じ発想です）。

> ⚠️ **`reshape_for_broadcast` は絶対に省略しないでください（実装で最も間違えやすい箇所）**
>
> `xq_` の形は `(バッチ, 系列長, ヘッド数, head_dim/2)` の4次元ですが、
> `freqs_cis` は `(系列長, head_dim/2)` の2次元です。そのまま掛けると
> PyTorch のブロードキャスト規則は**後ろの軸から**合わせようとするので、
> `系列長` が `ヘッド数`（この章の設定では 16）とぶつかり、
>
> ```
> RuntimeError: The size of tensor a (16) must match the size of tensor b (512) ...
> ```
>
> で必ず落ちます。`freqs_cis.view(1, T, 1, -1)` のように
> 「系列長の軸だけ残して他を 1 にする」整形が必須です。
> さらに怖いのは、形がたまたま一致して**エラーにならずに間違った位置情報が入る**ケースで、
> 「学習は進むのに性能が出ない」という最悪のバグになります。
> `assert` で形を検査しているのはそのためです。

> ⚠️ **実装の落とし穴（この教材の開発中に実際に踏んだバグ）**
>
> `freqs_cis` を `max_seq_len` 分ちょうどしか事前計算していないと、
> **生成時に学習文脈長を超えた瞬間、空のスライスを掴んでクラッシュ**します。
>
> ```
> AssertionError: freqs_cis (0, 16) が x (6, 1, 8, 16) と合いません
> ```
>
> 学習は `max_seq_len` で切り出すので絶対に超えませんが、**生成は普通に超えます**
> （64トークンで学習したモデルに300トークン書かせる、など）。
> RoPE は学習長を超えて外挿できるのが売りなので、テーブルは長めに確保します。
> この教材では本文の `Transformer`（5.1.7）も [`code/05_llama.py`](https://github.com/thirtypower/kgr-llm/blob/main/code/05_llama.py) も
> `max_seq_len × 8` を確保しています。
> **「学習では絶対通るのに生成だけで落ちる」パターンの典型例**として覚えておいてください。

---

### 5.1.4 Attention（GQA + RoPE + Flash Attention）

第2章の Multi-Head Attention に、GQA・RoPE・高速化を加えたものです。

```python
import math
import torch.nn.functional as F

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
        self.n_rep = self.n_local_heads // self.n_kv_heads   # 複製回数
        self.head_dim = args.dim // args.n_heads

        # Q は全ヘッド分、K/V は kv_heads 分だけ作る ← GQA の本体（下の解説を参照）
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # PyTorch 2.0 以降なら Flash Attention が使える
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            self.register_buffer("mask", torch.triu(mask, diagonal=1))

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # ① RoPE で位置情報を注入
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # ② GQA: K/V を Q のヘッド数まで複製
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq, xk, xv = (t.transpose(1, 2) for t in (xq, xk, xv))

        # ③ Attention 計算（causal=True で自動的に未来をマスク）
        if self.flash:
            output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        else:
            scores = xq @ xk.transpose(-2, -1) / math.sqrt(self.head_dim)
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            output = F.softmax(scores.float(), dim=-1).type_as(xq) @ xv

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
```

#### ★ GQA は「何が」減るのか（`repeat_kv` で複製し直すのはなぜ？）

このコードを読むと、必ずこう思います——
**「`wk` を小さくしたのに、`repeat_kv` で Q と同じヘッド数まで複製し直している。それでは減っていないのでは？」**

答えを先に書きます。**3つのうち、減るのは2つ、減らないのは1つ**です。

| | GQA で減るか | なぜ |
|---|---|---|
| **① パラメータ数**（`wk`, `wv` の重み） | ✅ **減る** | `wk` の出力が `n_kv_heads × head_dim` と小さくなるので、行列自体が小さい（16→8 なら半分） |
| **② KV キャッシュ**（推論時にメモリに貯める K/V） | ✅ **大きく減る** ← **これが本命** | **キャッシュに貯めるのは `repeat_kv` する前の小さい K/V** だから |
| **③ Attention の計算量** | ❌ **変わらない** | 複製後は MHA と同じ形（Q が16ヘッドなら K も16ヘッド）で計算するため |

**②が本命である理由**：

```
生成中、キャッシュに貯まっていくもの（例：dim=1024, head_dim=64, 4096トークン）

   MHA（16ヘッド）:  16ヘッド × 64 × 4096トークン × 2(K と V) = 8.4M 個の数値 / 層
   GQA( 8ヘッド):    8ヘッド × 64 × 4096トークン × 2         = 4.2M 個の数値 / 層  ← 半分
                     ↑ 貯めるのは「複製前」の 8 ヘッド分だけ
```

つまり `repeat_kv` は、**キャッシュから小さい K/V を取り出した"後"に、計算のためだけに一時的に膨らませる**処理です。
膨らませた結果はキャッシュに戻しません。だから**保存量だけが減る**わけです。

**この「膨らませる場所」が図の要点**です。

```mermaid
flowchart LR
    X["入力 x<br/>(B, T, dim)"]
    WQ["wq<br/><i>16ヘッド分</i>"]
    WKV["wk / wv<br/><i>8ヘッド分だけ</i><br/>★ここで①パラメータが減る"]
    CACHE["<b>KV キャッシュ</b><br/>8ヘッド分の K・V を貯める<br/>★ここで②メモリが減る（本命）"]
    REP["<b>repeat_kv</b><br/>8 → 16 ヘッドに複製<br/><i>計算のためだけの一時的な膨張。<br/>キャッシュには戻さない</i>"]
    ATT["Attention 計算<br/>Q16 × K16<br/>★ここは MHA と同じ量<br/>→ ③計算量は減らない"]

    X --> WQ --> ATT
    X --> WKV --> CACHE --> REP --> ATT

    classDef io fill:#e8f0fe,stroke:#4a7ec4,stroke-width:2px
    classDef win fill:#e9f7ef,stroke:#3d9970,stroke-width:3px
    classDef neu fill:#fffbe6,stroke:#d4a72c
    classDef same fill:#fdecea,stroke:#d64545
    class X,WQ io
    class WKV,CACHE win
    class REP neu
    class ATT same
```

> 🔑 **一言でまとめると**：
> **GQA は「保存するものを減らす」技術で、「計算を減らす」技術ではありません。**
> 生成が長くなるほど KV キャッシュはどんどん膨らみ、**長文生成ではモデル本体より大きくなる**ことすらあります。
> そこを半分・4分の1にできるので、長いコンテキストを扱う現代の LLM では必須になりました。
>
> 💡 だから GQA の効果が見えるのは「**長文**」「**推論**」のときだけで、
> この章の最後（「実測」節）で小規模な学習の性能差が出ないのは、これが理由です。

#### Flash Attention

**Flash Attention** は、Attention 行列を一度に全部メモリに置かずに
ブロックごとに計算する高速化アルゴリズムです。
数学的な結果は同じで、**メモリ使用量と速度が大幅に改善**します。
PyTorch 2.0 以降は `F.scaled_dot_product_attention` を呼ぶだけで自動的に使われます。

**何が問題で、何を変えたのか**を1枚にします。

```mermaid
flowchart TB
    subgraph NAIVE ["❌ 素朴な実装：T×T の表を丸ごとメモリに置く"]
        direction LR
        N1["QKᵀ を全部計算<br/>(T×T)"] --> N2["softmax<br/>(T×T)"] --> N3["×V"]
        N4["T=8192 なら 1ヘッドで<br/>8192² ＝ 6,700万マス。<br/>GPU の<b>高速な on-chip メモリ（SRAM）</b>には<br/>到底入らないので、<b>遅い HBM に書き出して<br/>読み直す</b>ことになる<br/><i>（学習では逆伝播用に層ごとに保持も必要）</i>"]
    end

    subgraph FLASH ["⭕ Flash Attention：ブロックに切って、その場で足し込む"]
        direction LR
        F1["Q・K・V を<br/>小さなブロックに切る"] --> F2["1ブロックずつ<br/>スコア→softmax→×V<br/><i>途中結果を持ち回りで更新<br/>（online softmax）</i>"] --> F3["T×T の表を<br/><b>一度も作らない</b>"]
    end

    NAIVE --> KEY["<b>結果の数値は完全に同一。</b><br/>変えたのは「計算の順番と置き場所」だけ<br/>→ 保持するメモリが T² から T に、<br/>　 遅い HBM への往復が激減して速くなる"]
    FLASH --> KEY

    classDef bad fill:#fdecea,stroke:#d64545
    classDef good fill:#e9f7ef,stroke:#3d9970,stroke-width:2px
    classDef note fill:#fffbe6,stroke:#d4a72c,color:#665
    class N1,N2,N3 bad
    class F1,F2,F3 good
    class N4,KEY note
```

> 🔑 **Flash Attention は「近似」ではありません。**
> 出力は素朴な実装と（丸め誤差の範囲で）一致します。
> **速くなる理由は計算量ではなくメモリ移動**——GPU の遅い側のメモリ（HBM）と
> 速い側（SRAM）の往復を減らしただけ、というのがこの手法の面白いところです。
> 逆に言うと **計算量（T² のオーダー）は減っていない**ので、
> 長文化の本質的な解決にはなりません（→ [05.5章 5.5.4](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/055-moe-modern-arch)）。

---

### 5.1.5 MLP（SwiGLU）

LLaMA の FFN は、第2章の 2層 ReLU ではなく **3つの線形層 + SiLU** を使います。

```mermaid
flowchart LR
    X["x"] --> W1["w1"] --> SILU["SiLU"]
    X --> W3["w3"]

    SILU --> MUL(("×"))
    W3 --> MUL
    MUL --> W2["w2"] --> OUT["出力"]

    NOTE["要素ごとの掛け算<br/>＝ <b>ゲート機構</b><br/>w3 が「どの情報を通すか」の<br/>フィルタとして働く"]
    NOTE -.- MUL

    classDef lin fill:#e8f0fe,stroke:#4a7ec4,stroke-width:2px
    classDef act fill:#fffbe6,stroke:#d4a72c
    classDef note fill:#fffbe6,stroke:#d4a72c,color:#665
    classDef io fill:#f6f6f6,stroke:#999
    class W1,W2,W3 lin
    class SILU act
    class NOTE note
    class X,OUT io
```

```python
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)          # SwiGLU 用に 2/3 に縮める
            # multiple_of の倍数に切り上げ（GPU 効率のため）
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

#### 電卓だけで確かめる：「ゲート」は何をしているのか

「`w3(x)` がフィルタとして働く」と言われてもピンと来ないので、**数値を4個だけ**通します。
`hidden_dim = 4` として、2つの線形層の出力がこうなったとします。

```
   a = w1(x)  =  [ 2.0 ,  −3.0 ,   0.5 ,   1.0 ]     ← 「特徴の値」
   g = w3(x)  =  [ 1.5 ,   1.0 ,   0.0 ,  −1.0 ]     ← 「ゲート（通す量）」
```

| | 1番目 | 2番目 | 3番目 | 4番目 |
|---|---|---|---|---|
| `a`（特徴） | 2.0 | −3.0 | 0.5 | 1.0 |
| `SiLU(a)` | 1.7616 | **−0.1423** | 0.3112 | 0.7311 |
| `g`（ゲート） | 1.5 | 1.0 | **0.0** | **−1.0** |
| **`SiLU(a) × g`** → w2 へ | **2.6424**<br/>1.5倍に増幅 | −0.1423<br/>そのまま通過 | **0.0000**<br/>**完全に遮断** | **−0.7311**<br/>**符号が反転** |

**ここが ReLU の FFN との決定的な違い**です。

| | 2層 ReLU の FFN（02.5章） | SwiGLU |
|---|---|---|
| その特徴を通すかどうかを決めるのは | **その特徴自身の値**（負なら 0、正ならそのまま） | **もう1本の別の射影 `w3(x)`** |
| 同じ特徴値 0.5 の扱い | 常に `0.5` が通る | `SiLU(0.5)=0.311` に `g` を掛けるので、`g=0` なら 0、`g=2` なら 0.62 |
| 言い換えると | 「大きければ通る」 | **「今の入力の文脈では、この特徴を使うべきか？」を別途判断する** |

> 🔑 **ゲートとは「掛け算で作るスイッチ」**です。
> 足し算では情報を消せませんが、**0 を掛ければ確実に消えます**。
> 「同じ特徴でも、文脈によって使う／使わないを切り替えられる」——
> これが線形層1本ぶんのコストを払ってでも SwiGLU が使われる理由です。
>
> 💡 [05.5章の MoE](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/055-moe-modern-arch) の「ルータ」も、
> 発想はこれと同じ「掛け算で選ぶ」の拡張版です
> （あちらは FFN まるごとを選ぶ、という粒度の違い）。

**SiLU（Swish）**：`SiLU(x) = x · sigmoid(x)`。ReLU と違って滑らかで、負の値も少し通します。

| x | −3.0 | −1.0 | −0.5 | 0.0 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|---|
| **SiLU(x)** | −0.142 | **−0.269** | −0.189 | 0.000 | 0.311 | 0.731 | 1.762 |
| ReLU(x) | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 1.00 | 2.00 |

> 📝 **負の側が「少しだけ」通るのがポイント**です。
> ReLU は負の入力で出力も勾配も完全に 0 になるため、
> そのニューロンが**二度と学習しなくなる**ことがあります（dying ReLU）。
> SiLU は −0.27 付近を底に持つ滑らかな曲線なので、勾配が残ります。

#### `hidden_dim` はなぜ 2752 のような中途半端な数になるのか

線形層が1つ増える分、`hidden_dim` を 2/3 に縮めてパラメータ数を揃えています。
コードの3行を `dim = 1024` で実際に計算すると：

```
   ① hidden_dim = 4 × dim         = 4 × 1024      = 4096
   ② hidden_dim = int(2/3 × 4096)                 = 2730     ← 3本ぶんに合わせて縮める
   ③ 64 の倍数に切り上げ            2730 → 2752            ← GPU が扱いやすい形に
```

- **なぜ 2/3 か**：2層 FFN のパラメータは `dim × 4dim × 2` 本ぶん。
  3層にすると `dim × h × 3` 本になるので、`h = 4dim × 2/3` にすれば **合計がほぼ同じ**になります
- **なぜ 64 の倍数か**：行列積は決まった大きさのブロック単位で計算されるため、
  端数があると最後のブロックが無駄になります（`multiple_of=64` の意味）

---

### 5.1.6 Decoder Layer

部品を組み立てます。**Pre-Norm 構成**（正規化が各サブ層の前）です。

```python
class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis):
        # 残差接続： x + サブ層(Norm(x))
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
```

たった6行ですが、ここに **第2章で学んだ Attention・FFN・正規化・残差接続がすべて**入っています。

```mermaid
flowchart TD
    X["x"]
    X --> N1["RMSNorm"] --> ATT["<b>Attention</b><br/>GQA ＋ RoPE"]
    X -- "残差接続" --> ADD1(("＋"))
    ATT --> ADD1
    ADD1 --> H["h"]

    H --> N2["RMSNorm"] --> MLP["<b>MLP</b><br/>SwiGLU"]
    H -- "残差接続" --> ADD2(("＋"))
    MLP --> ADD2
    ADD2 --> OUT["out"]

    classDef main fill:#e9f7ef,stroke:#3d9970,stroke-width:2px
    classDef norm fill:#f0f0f0,stroke:#888
    classDef io fill:#f6f6f6,stroke:#999
    class ATT,MLP main
    class N1,N2 norm
    class X,H,OUT io
```

> 💡 [02.7章](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/027-mini-gpt) の GPT-2 のブロック図と**見比べてみてください**。
> 形は完全に同じで、**箱の中身が LayerNorm→RMSNorm、MHA→GQA+RoPE、ReLU FFN→SwiGLU
> に入れ替わっているだけ**だと分かります。

---

### 5.1.7 モデル全体

```python
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

        # 重み共有（Weight Tying）：入力の Embedding と出力層で同じ重みを使う
        self.tok_embeddings.weight = self.output.weight

        # RoPE の回転角を事前計算してバッファに保持
        # ★ max_seq_len「ちょうど」ではなく 8 倍まで確保する（5.1.3 の警告参照）。
        #   学習は max_seq_len で切り出すので超えないが、生成は普通に超える。
        #   テーブルが足りないと生成の途中で空のスライスを掴んで落ちる。
        freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 8)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, tokens, targets=None):
        _bsz, seqlen = tokens.shape
        h = self.dropout(self.tok_embeddings(tokens))
        freqs_cis = self.freqs_cis[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cis)
        h = self.norm(h)

        if targets is not None:
            # 学習時：全位置で損失を計算
            logits = self.output(h)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
            return logits, loss
        else:
            # 推論時：最後の位置だけ計算すれば十分（高速化）
            logits = self.output(h[:, [-1], :])
            return logits, None
```

**重み共有（Weight Tying）**：入力 Embedding（語彙数×次元）と出力層（次元×語彙数）は
形が転置の関係にあるため、同じ行列を使い回せます。パラメータを大幅に節約でき、性能も落ちません。

> ⚠️ **この本文版には KV キャッシュが入っていません（意図的な省略です）**
>
> [02.7章の検査5](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/027-mini-gpt) で「KV キャッシュは実務では必須」と書いたのに、
> 上の `forward` には `cache` 引数がありません。理由は、**本文版は「構造を理解して写経する」ため**の
> 最小構成にしてあるからです（学習には KV キャッシュを一切使いません。使うのは生成時だけ）。
>
> **生成の高速化まで含んだ完成版は [`code/05_llama.py`](https://github.com/thirtypower/kgr-llm/blob/main/code/05_llama.py) 側**にあり、
> `forward(tokens, targets=None, caches=None, start_pos=0)` という形で2つの引数が増えています。
>
> ```python
> # code/05_llama.py 側（イメージ）
> def forward(self, tokens, targets=None, caches=None, start_pos: int = 0):
>     ...
>     # ★ ここが本文版との決定的な違い
>     freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
> ```
>
> 🔑 **`start_pos` が要る理由（KV キャッシュ実装の最大の落とし穴）**：
> キャッシュを使う生成では、**毎回「新しい1トークンだけ」を forward に渡します**。
> すると `seqlen` は常に 1 なので、本文版の `self.freqs_cis[:seqlen]` は
> **いつも「位置0の回転角」を拾ってしまいます**。
> 500トークン目を処理していても「これは1個目です」と嘘の位置情報を入れることになり、
> **エラーは出ないのに生成が崩壊する**——という最悪のバグになります。
> だから「いま何トークン目から処理しているか」を `start_pos` で渡す必要があります。
>
> 02.7章の検査5（**一括 forward の結果と、キャッシュを使った1トークンずつの結果が一致するか**）は、
> まさにこのズレを検出するための検査です。

これで **`dim=768, n_layers=12, n_heads=16` なら約 8259万パラメータ**、
**`dim=1024, n_layers=18` なら約 2.15億パラメータ**のモデルができました。

#### ★ 「215M」の中身を1つずつ数える

この「2.15億」がどこから来ているのかを、**実際に数え上げます**
（`dim=1024, n_layers=18, n_heads=16, n_kv_heads=8, vocab=6144, hidden_dim=2752`）。
`torch` で数えた値と、電卓で出せる式を並べます。

| 部品 | 数え方 | 1層あたり | 全体（×18層） | 割合 |
|---|---|---|---|---|
| **Embedding**（`tok_embeddings`） | `vocab × dim` = 6144 × 1024 | — | **6,291,456** | 2.9% |
| **Attention** `wq` | `dim × (n_heads × head_dim)` = 1024 × 1024 | 1,048,576 | | |
| **Attention** `wk` | `dim × (n_kv_heads × head_dim)` = 1024 × **512** | 524,288 | | |
| **Attention** `wv` | 同上 | 524,288 | | |
| **Attention** `wo` | `dim × dim` | 1,048,576 | | |
| **Attention 小計** | | **3,145,728** | **56,623,104** | 26.3% |
| **FFN** `w1` / `w3` | `dim × hidden_dim` = 1024 × 2752 | 2,818,048 ×2 | | |
| **FFN** `w2` | `hidden_dim × dim` | 2,818,048 | | |
| **FFN 小計** | | **8,454,144** | **152,174,592** | **70.7%** |
| **RMSNorm** | `dim` × 2本／層 ＋ 最終1本 | 2,048 | 37,888 | 0.02% |
| **出力層** | **Embedding と共有**（Weight Tying） | — | **0** | 0% |
| **合計** | | | **215,127,040** | 100% |

**この表から持ち帰るべきことが4つあります。**

| 気づき | 意味 |
|---|---|
| **FFN が 7割** | [02.5章 ⑦](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/025-transformer-block) の「パラメータの 2/3 は FFN」がそのまま出ています。**LLM の"知識"はほぼ FFN にある**、と言われる根拠です。だから[05.5章の MoE](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/055-moe-modern-arch) は FFN だけを増やします |
| **`wk`/`wv` が `wq` の半分** | GQA（`n_kv_heads=8`）の効果です。MHA なら Attention 小計は 4,194,304／層 ＝ 全体 75.5M で、**約 19M 増えます** |
| **Embedding が 2.9% しかない** | 語彙 6144 が小さいためです。実物（vocab=128k、LLaMA-3）では **Embedding だけで数億**になり、小型モデルでは無視できない比率になります |
| **出力層が 0** | Weight Tying（入口と出口で同じ行列を使う）のおかげ。切ると **+6.3M** です |

> 💡 **この数え方が使えると、実務で効きます。**
> 「このモデルは VRAM に載るか？」は
> **パラメータ数 × バイト数（bf16 なら2）** が出発点です。
> 215M なら重みだけで 430MB、学習するなら [06章 6.1.5](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/06-training-pipeline) の
> 16バイト/パラメータで **約 3.4GB** ——という見積もりが即座にできます。

---

## 5.2 トークナイザを訓練する

### 5.2.1 トークナイザの種類

| 方式 | 例 | 長所 | 短所 |
|------|-----|------|------|
| **単語単位** | `["I", "love", "cats"]` | 直感的 | 語彙が爆発、未知語に弱い |
| **文字単位** | `["I", "l", "o", "v", "e"]` | 語彙が小さい、未知語なし | 系列が長くなる、意味を持たない |
| **サブワード** | `["I", "lov", "e", "cat", "s"]` | **両者の良いとこ取り** | 訓練が必要 |

現代の LLM は全て **サブワード方式**です。

### 5.2.2 BPE の仕組み

**BPE（Byte Pair Encoding）** は、「**最も頻繁に隣り合う2つの記号を、1つにまとめる**」
という操作を、目標の語彙数に達するまで繰り返します。

```
初期状態（文字単位に分解）：
  l o w </w>            ×5
  l o w e r </w>        ×2
  n e w e s t </w>      ×6
  w i d e s t </w>      ×3

Step 1: 最頻ペアは "e s"（6+3=9回）→ 結合
  n e w es t </w>       w i d es t </w>

Step 2: 最頻ペアは "es t"（9回）→ 結合
  n e w est </w>        w i d est </w>

Step 3: "est </w>" → 結合
  n e w est</w>         w i d est</w>

...これを語彙数に達するまで繰り返す
```

こうして、頻出する単語は1トークンに、稀な単語は複数の断片になります。

### 5.2.3 実際に訓練する

Hugging Face の `tokenizers` ライブラリを使います。

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
import json

# ① BPE モデルで初期化
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# ② 特殊トークンを定義
special_tokens = ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"]

# ③ トレーナーを設定
trainer = trainers.BpeTrainer(
    vocab_size=6144,
    special_tokens=special_tokens,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

# ④ コーパスを流し込んで訓練
def read_texts(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)['text']

tokenizer.train_from_iterator(read_texts("corpus.jsonl"), trainer=trainer)

import os
os.makedirs("./my_tokenizer", exist_ok=True)   # 保存先フォルダを先に作る（無いと save が例外になる）
tokenizer.save("./my_tokenizer/tokenizer.json")
```

#### 特殊トークンの役割

| トークン | 役割 |
|---------|------|
| `<unk>` | 未知語（Byte-Level BPE ならほぼ出ない） |
| `<s>` / `</s>` | 文書の開始 / 終了 |
| `<\|im_start\|>` / `<\|im_end\|>` | **会話の役割の境界**（SFT で使用） |

#### Chat Template の設定

Hugging Face 形式で保存するとき、会話のフォーマットを Jinja2 テンプレートで指定します。

```jinja
{% for message in messages %}
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
```

これにより、`tokenizer.apply_chat_template(messages)` で自動整形できるようになります。

**このテンプレートはどこに保存されるのか**：Hugging Face 形式では、
`tokenizer.json`（語彙とマージ規則）と同じフォルダの **`tokenizer_config.json` の
`chat_template` フィールド**に文字列として入ります。最小の設定例：

```json
// my_tokenizer/tokenizer_config.json
{
  "tokenizer_class": "PreTrainedTokenizerFast",
  "bos_token": "<s>",
  "eos_token": "</s>",
  "unk_token": "<unk>",
  "chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
}
```

Python からは `tokenizer.chat_template = "..."` を代入してから
`tokenizer.save_pretrained("./my_tokenizer")` すれば、このファイルが自動生成されます。

#### 語彙サイズの選び方

| 語彙数 | トレードオフ |
|-------|------------|
| 小さい（6K） | Embedding 層が軽い。ただし1文あたりのトークン数が増える |
| 大きい（128K） | 文が短いトークン列で表せる。ただし Embedding が巨大に |

本教材が 6144 と小さいのは、モデル自体が小さいためです。
（LLaMA-2 は 32000、LLaMA-3 は 128256）

---

## 5.3 小さな LLM を事前学習する

### 5.3.1 データの準備

本文で例として挙げるデータ：

| 用途 | データセット | 規模 |
|------|------------|------|
| 事前学習 | `wikimedia/wikipedia`（`20231101.ja`）、[llm-jp-corpus](https://gitlab.llm-jp.nii.ac.jp/) | 日本語版 Wikipedia だけで 130万記事以上（絞って使う） |
| SFT | `kunishou/databricks-dolly-15k-ja`、`llm-jp/oasst1-21k-ja` | 1.5万〜2万件の指示-応答 |

#### データを実際にダウンロードして corpus.jsonl を作る

以降のコードは、**1行1文書の JSONL 形式**（`{"text": "..."}`）のファイルを前提にします。
Hugging Face Hub から `datasets` ライブラリで日本語 Wikipedia を落として変換する例です。

```python
# pip install datasets
from datasets import load_dataset
import json

N = 50_000    # まずは5万記事だけ（日本語版全体は130万記事以上あるので絞る）

# streaming=True が重要：全量（数GB）をダウンロードせず、先頭から順に読める
ds = load_dataset("wikimedia/wikipedia", "20231101.ja",
                  split="train", streaming=True)

with open("corpus.jsonl", "w", encoding="utf-8") as f:
    for i, row in enumerate(ds):
        if i >= N:
            break
        text = row["text"].strip()
        if len(text) >= 200:    # 極端に短い記事（リダイレクト等）は除外
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
```

- 5万記事で**約 1.5億文字 ≒ 0.1B トークン前後**。まず1晩の学習にはこのくらいが手頃です
- 本気で回すなら `N` を増やすだけ。Chinchilla 則（215M × 20 ≒ 4.3B トークン）が上の目安
- この手順を含む完全版が [`code/06_pretrain.py`](https://github.com/thirtypower/kgr-llm/blob/main/code/06_pretrain.py) の `--full` モードです

### 5.3.2 Dataset クラス

**事前学習用**：テキストをトークン化して、固定長に切り分けるだけ。

```python
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 全テキストをトークン化して1本の長い列に連結
        # ※ load_and_tokenize の中身はここでは省略。完全版は code/06_pretrain.py
        #   （corpus.jsonl を1行ずつ読み、tokenizer.encode して </s> 区切りで連結するだけ）
        self.data = self.load_and_tokenize(data_path)

    def __len__(self):
        return len(self.data) // self.max_length

    def __getitem__(self, index):
        start = index * self.max_length
        chunk = self.data[start : start + self.max_length + 1]
        X = torch.tensor(chunk[:-1], dtype=torch.long)   # 入力
        Y = torch.tensor(chunk[1:],  dtype=torch.long)   # 正解（1つずらし）
        return X, Y
```

**ポイント**：`Y` は `X` を1つ左にずらしたもの。これが「次トークン予測」の実装です。

#### ★ 実物のバッチを1件だけ覗く（実測値）

説明だけでは形が掴めないので、**`code/tokenizer_demo.json`（語彙346）で実際に通した値**を出します。
`max_length = 8`、テキストは `"猫が魚を食べる。太郎は公園へ行く。"` の1文です。

```
   トークン化した結果（14トークン）
     [312, 266, 321, 265, 275, 277, 264, 305, 268, 327, 280, 282, 283, 264]
      猫    が   魚    を   食   べる   。   太郎   は   公園   へ    行    く    。
```

`__len__` は `14 // 8 = 1`。つまり**この文からは1件しか作られません**。

| | 位置0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| **`X`（入力）** | 312<br/>猫 | 266<br/>が | 321<br/>魚 | 265<br/>を | 275<br/>食 | 277<br/>べる | 264<br/>。 | 305<br/>太郎 |
| **`Y`（正解）** | 266<br/>が | 321<br/>魚 | 265<br/>を | 275<br/>食 | 277<br/>べる | 264<br/>。 | 305<br/>太郎 | 268<br/>は |

**縦に読むと「この入力の次はこれ」**という8個のクイズになっているのが分かります
（位置0 では「猫」を見て「が」を当てる、位置7 では「猫が魚を食べる。太郎」まで見て「は」を当てる）。

> ⚠️ **`chunk` が `max_length + 1` 個ぶん読んでいる理由**がここで分かります。
> `X` に8個、`Y` に8個必要なので、**重なりを考えると9個**要ります。
> ここを `max_length` にすると、`Y` が1個足りずに落ちます（自作時の定番バグ）。
>
> 📝 **9番目以降の5トークン（公園／へ／行／く／。）は一度も使われません。**
> `len(data) // max_length` が端数を切り捨てるからです。
> 実データでは全体の一部なので気にしませんが、**「データが少ないのに loss が下がらない」**
> ときは、ここで大量に捨てていないか確認してください
> （実装によっては、文をぎっしり詰めて無駄を出さない **packing** を使います）。

**SFT 用**：損失マスクが加わります。

その前に1つ準備が必要です。5.2.3 で作ったのは `tokenizers` ライブラリの
`Tokenizer` オブジェクトで、これは `tokenizer(prompt).input_ids` という
**呼び出し方に対応していません**（そのまま書くと `TypeError` になります）。
`transformers` の `PreTrainedTokenizerFast` でラップすると、
Hugging Face 標準の呼び出し方（`__call__`・`pad_token_id`・`apply_chat_template` 等）が使えるようになります。

```python
# pip install transformers
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="./my_tokenizer/tokenizer.json",   # 5.2.3 で保存したファイル
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="</s>",    # 専用の <pad> を作らず </s> を流用（パディングは損失計算外なので問題ない）
)
print(tokenizer("こんにちは").input_ids)   # ← これが動くようになる
```

```python
import json

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # SFT データは jsonl（1行 = {"instruction": 質問, "output": 回答}）
        with open(data_path, encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # Chat Template で整形
        prompt = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n"
        answer = f"{item['output']}<|im_end|>"

        prompt_ids = self.tokenizer(prompt).input_ids
        answer_ids = self.tokenizer(answer).input_ids
        input_ids = prompt_ids + answer_ids

        # ★ 質問部分のラベルを -100 にして損失計算から除外
        labels = [-100] * len(prompt_ids) + answer_ids

        # 長すぎる分は切り詰め、足りない分は pad_token_id で埋めて固定長に
        # （固定長にしないと DataLoader が batch にまとめられない）
        input_ids = input_ids[:self.max_length]
        labels    = labels[:self.max_length]
        pad_len   = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels    = labels    + [-100] * pad_len   # パディング部分も損失計算から除外
        return torch.tensor(input_ids), torch.tensor(labels)
```

`-100` は PyTorch の `cross_entropy` のデフォルト `ignore_index` です。
→ 第4章で説明した「損失マスク」の実装がこれです。

#### ★ SFT のバッチも実物で見る（実測値）

同じトークナイザで、`max_length = 20` の1件を作ります。
（検証用コーパスに会話文が無いので instruction / output は例文2つで代用し、
表を短くするため Chat Template の `user` / `assistant` の文字も省いています——
この語彙では `user` が4トークン、`assistant` が9トークンになるので、
**実物では役割名だけで13トークン消えます**。それも含めて「枠の使われ方」を見てください。）

```
   instruction = "猫は魚を食べる。"      → prompt_ids  9個（<|im_start|> と <|im_end|> を含む）
   output      = "花子は学校へ行く。"     → answer_ids  8個（<|im_end|> を含む）
```

| 位置 | 0 | 1〜7 | 8 | 9〜15 | 16 | 17〜19 |
|---|---|---|---|---|---|---|
| 中身 | `<\|im_start\|>` | 猫 は 魚 を 食 べる 。 | `<\|im_end\|>` | 花子 は 学校 へ 行 く 。 | `<\|im_end\|>` | `</s>`（パディング） |
| **`input_ids`** | 3 | 312 268 321 265 275 277 264 | 4 | 296 268 330 280 282 283 264 | 4 | 2 2 2 |
| **`labels`** | **−100** | **−100 ×7** | **−100** | 296 268 330 280 282 283 264 | 4 | **−100 ×3** |
| 損失に効くか | ❌ | ❌ | ❌ | ⭕ | ⭕ | ❌ |

> 🔑 **20枠のうち、実際に学習しているのは 8枠だけ**（40%）です。
> 残りは「読むけれど真似しない」質問文と、「何も無い」パディングです。
>
> | 何を数えるか | 個数 |
> |---|---|
> | 確保したメモリ（`max_length`） | 20 |
> | 実際に中身のあるトークン | 17 |
> | **損失が計算される位置** | **8** |
>
> 📝 **このコードが `attention_mask` を返していない**ことに気づいた人へ。
> 手抜きではなく、**この形なら要らない**からです。
> ①Attention は因果的（未来を見ない）なので、**末尾**のパディングは
> それより前の位置の計算に一切影響しません。
> ②パディング位置自身の出力は `labels = -100` で損失から外れています。
> ——つまり「**右詰めパディング ＋ 損失マスク**」の組み合わせで辻褄が合っています。
>
> ⚠️ ただし **左詰めパディング（left padding）では成立しません**。
> `code/05_llama.py` のモデルに、同じ6トークンを①そのまま ②右に6個パディング
> ③左に6個パディングして通し、logits を比べた実測です。
>
> ```
>    右詰め : 同じ6位置の logits の最大差 = 0.00000015   ← 実質ゼロ（誤差）
>    左詰め : 同じ6トークンの logits の最大差 = 0.375     ← 完全に別物
> ```
>
> 左詰めが壊れる理由は2つあります。**①本文の各位置が前のパディングを見てしまう**、
> **②RoPE の位置が6つずれる**。
> 生成時（`generate`）はバッチ内の長さを揃えるために左詰めにするのが定石なので、
> **そのときは `attention_mask`（と正しい位置）が必須**です。
> **「学習は右詰め、生成は左詰め」**と覚えて、左詰めのときだけ気をつけてください。
>
> 📝 だから **SFT は「見た目のトークン数」よりずっと学習が進みません**。
> `max_length` を大きく取りすぎると、パディングだけの計算に GPU 時間を払うことになります。
> 実務では **長さの近いサンプルを同じバッチに集める**（length grouping）か、
> **複数サンプルを詰め込む**（packing）ことで、この 40% を上げます。
>
> ⚠️ **`<|im_end|>`（位置16）が学習対象に入っている**ことに注目してください。
> ここを `-100` にすると、モデルは**話し終わる合図を学ばず、延々と書き続けます**。
> 「生成が止まらない」バグの典型的な原因です。

### 5.3.3 学習ループ

```python
import math
from torch.utils.data import DataLoader

def get_lr(step, total_steps, warmup_steps, lr_max, lr_min=0.0):
    """学習率スケジューラ：ウォームアップ + コサイン減衰"""
    if step < warmup_steps:
        return lr_max * step / warmup_steps                    # 線形に上げる
    ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * ratio))

model = Transformer(ModelConfig(dim=1024, n_layers=18)).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1,
                              betas=(0.9, 0.95))
# ※ PretrainDataset(...) の引数は 5.3.2 参照。完全版は code/06_pretrain.py
loader = DataLoader(PretrainDataset("corpus.jsonl", tokenizer), batch_size=32, shuffle=True)

ACCUM_STEPS = 8                                # 勾配累積
EPOCHS = 1                                     # 事前学習は 1 エポックが基本（データを増やす方が効く）
TOTAL_STEPS = EPOCHS * len(loader) // ACCUM_STEPS   # optimizer.step() が呼ばれる総回数
WARMUP = max(1, int(TOTAL_STEPS * 0.03))       # 全体の 3% をウォームアップに充てる

# 混合精度学習用。bf16 が使える GPU（RTX 30 系以降）なら fp16 より bf16 を推奨
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

step = 0
for epoch in range(EPOCHS):
    for i, (X, Y) in enumerate(loader):
        X, Y = X.cuda(), Y.cuda()

        # 学習率を更新
        lr = get_lr(step, TOTAL_STEPS, WARMUP, 3e-4)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # 混合精度で forward
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits, loss = model(X, targets=Y)
            loss = loss / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 勾配クリッピング
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

# 学習が終わったら重みを保存する（5.3.4 の SFT はこのファイルから始まる）
torch.save(model.state_dict(), "pretrain.pth")
```

> 📘 **`GradScaler` とは何か（fp16 のための損失スケーリング）**
>
> fp16（float16）は**表現できる数の範囲が狭く**、逆伝播で出てくる小さな勾配が
> 0 に潰れてしまうことがあります（**アンダーフロー**）。それを防ぐのが `GradScaler` です。
> やっていることは単純で、**損失を一時的に大きな倍率（例：65536倍）で拡大してから
> backward し、重みを更新する直前に元の倍率に戻す**だけです。
> 勾配全体が定数倍されるだけなので、学習の数学的な意味は変わりません。
>
> 前章までの素朴なループとの対応：
>
> | 素朴なループ | GradScaler 版 | 何をしている |
> |------------|--------------|-------------|
> | `loss.backward()` | `scaler.scale(loss).backward()` | 損失を拡大してから逆伝播 |
> | （なし） | `scaler.unscale_(optimizer)` | 勾配を元の倍率に戻す（クリッピングの**前**に必要） |
> | `optimizer.step()` | `scaler.step(optimizer)` | 勾配が発散（inf/NaN）していたら更新をスキップ |
> | （なし） | `scaler.update()` | 拡大倍率を自動調整（発散したら下げる） |
> | `optimizer.zero_grad()` | `optimizer.zero_grad()` | 同じ |
>
> **bf16 なら不要**です。bf16 は fp32 と同じ指数部を持ち、表現範囲が広いので
> アンダーフローがほぼ起きません。上のコードで `enabled=(amp_dtype == torch.float16)`
> としているのはそのためで、bf16 のときの `scaler` は「何もしない素通し」になります。

#### 学習テクニックの解説

| テクニック | 何のため |
|-----------|---------|
| **ウォームアップ** | 最初から高い学習率だと発散する。徐々に上げる |
| **コサイン減衰** | 終盤に学習率を下げて、細かく収束させる |
| **勾配累積（Gradient Accumulation）** | GPU メモリが足りなくても、実質的に大きいバッチサイズを実現 |
| **混合精度（AMP / bf16）** | float32 の代わりに bfloat16 を使い、メモリ半分・速度2倍 |
| **勾配クリッピング** | 勾配が急に大きくなったときに上限で切り、学習の暴走を防ぐ |
| **AdamW** | Adam に weight decay を正しく適用したもの。LLM の標準 optimizer |

**上の3つ（スケジュール・累積・クリッピング）は数字を1回見れば終わります。**

#### ① 学習率スケジュール：実際にどんな値が流れるのか

`get_lr()` に `TOTAL_STEPS=1000, WARMUP=30, lr_max=3e-4` を入れた**実際の出力**です。

| step | 0 | 1 | 10 | 20 | **30** | 100 | 300 | 500 | 700 | 900 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **lr** | 0 | 1e-5 | 1e-4 | 2e-4 | **3e-4** | 2.96e-4 | 2.46e-4 | 1.57e-4 | 6.5e-5 | 7.8e-6 | 0 |
| lr_max 比 | 0% | 3% | 33% | 67% | **100%** | 99% | 82% | **52%** | 22% | 2.6% | 0% |

```
 3e-4 (100%) |  *********
       (83%) |           ********
       (67%) | *                 *****
       (50%) |                        *****
       (33%) |                             ******
       (17%) |                                   *******
       0     +*                                         **********
             +----------------------------------------------------
              0  30                                          1000  step
              └warmup┘└─────────── cosine 減衰 ───────────┘
```

| 気づき | 意味 |
|---|---|
| **step 0 の lr は 0** | 最初の1回は**まったく更新しません**。「ランダムな重みに大きな一歩」を踏ませないための保険です |
| ウォームアップは全体の **3%** | 1000ステップなら30。ここが短すぎると初期に発散し、長すぎると単に遅くなります |
| 折り返し（step 500）で **52%** | コサインは中盤までゆっくり、終盤で急に落ちます。**学習の大半を高い lr で走る**設計です |
| 最後は **ほぼ 0** | 終盤に細かく詰めるため。途中で学習を止めると「まだ lr が高い状態の重み」になるので、**性能が出ません**（途中終了した checkpoint が期待より弱いのはこれが原因） |

#### ② 勾配累積：なぜ `loss / ACCUM_STEPS` で割るのか

`batch_size=32, ACCUM_STEPS=8` なら **実効バッチ ＝ 32 × 8 ＝ 256** です。
ここで割り算を忘れると何が起きるかを、数字で見ます。

```
   8回ぶんの micro-batch の loss
     2.1 , 2.3 , 1.9 , 2.0 , 2.2 , 2.4 , 1.8 , 2.1

   PyTorch の backward は勾配を「足し込む」（上書きしない）ので……

   ❌ 割らない場合 :  合計 16.8 に対応する勾配が溜まる
                      → 平均の 8倍の大きさ ＝ 学習率を勝手に 8倍したのと同じ
   ⭕ 8 で割る場合 :  平均 2.1 に対応する勾配が溜まる
                      → 「バッチ256 で1回 backward した」のと（ほぼ）同じ
```

> 🔑 **勾配累積は「メモリを時間で買う」テクニック**です。
> バッチ256 を一度に載せられない GPU でも、**32 ずつ8回に分けて同じ結果**が得られます。
> 代償は時間で、8回ぶんの forward/backward が必要です。
>
> ⚠️ **`optimizer.zero_grad()` を毎回呼んではいけません**（累積が消えます）。
> 上のコードで `if (i+1) % ACCUM_STEPS == 0:` の中にあるのはそのためです。
> 逆に [00.5章](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/005-deep-learning-basics) の素朴なループで毎回ゼロにしていたのは、
> 「1バッチ ＝ 1更新」だったからです。**同じ `zero_grad` が、置き場所で意味が変わります。**

> ⚠️ **「（ほぼ）同じ」と書いた理由**——ここは実際にライブラリ側のバグになった箇所です。
> `ACCUM_STEPS` で割る方法が**厳密に等しくなるのは、micro-batch ごとの
> 「損失を計算するトークン数」が全部同じときだけ**です。
>
> ```
>    micro-batch A : 損失対象トークン 100個 の平均 loss
>    micro-batch B : 損失対象トークン  10個 の平均 loss
>
>    ❌ 単純に (A + B) / 2       → 10個しかない B が、100個の A と同じ重みを持つ
>    ⭕ (Aの合計 + Bの合計) / 110 → バッチ110で1回やったのと一致
>  ```
>
> **事前学習では固定長なので差は出ません**が、**SFT では応答の長さがバラバラ**なので
> 差が出ます（短い応答が過大評価される）。
> 2024年に Hugging Face 側でこの正規化が修正され、いまの `Trainer` は
> **累積したトークン数で割る**実装になっています。
> 自分で学習ループを書くときは、**「何で割っているか」を必ず確認してください**
> ——[08章 8.1.4](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/08-reinforcement-learning) の GRPO の「損失の集計単位」と、まったく同じ種類の落とし穴です。

#### ③ 勾配クリッピング：何を「切る」のか

切るのは**個々の勾配の値ではなく、全パラメータをまとめた1本のベクトルの長さ**です。
2次元で書くと、こうです。

```
   勾配 g = [ 3.0 , 4.0 ]          長さ |g| = √(3²+4²) = 5.0
   max_norm = 1.0

   5.0 > 1.0 なので、全体を  1.0 / 5.0 = 0.2 倍する
     → g' = [ 0.6 , 0.8 ]          長さ |g'| = 1.0  ✅

   ★ 向きは変わっていない（3:4 の比のまま）
```

| | 何が保たれるか |
|---|---|
| **向き（どちらに動くか）** | ⭕ 完全に保たれる。比率を一切変えない |
| **大きさ（どれだけ動くか）** | ❌ 上限 1.0 に抑えられる |

> 🔑 **「進む方向は信じるが、歩幅は信じない」**という判断です。
> 学習中に一度だけ現れる異常なバッチ（壊れたテキスト、極端に長い文）で
> 勾配が 100 倍になることがあり、それが**1回でも通ると重みが壊れます**。
> `clip_grad_norm_(..., 1.0)` は LLM 学習の**ほぼ全ての実装に入っている**定番です。
>
> 📝 `clip_grad_norm_` は**切った後の長さ**ではなく**切る前の長さ**を返します。
> これをログに出すと、`grad_norm` が普段 0.3 なのに突然 40 になる瞬間が見えます——
> **データ品質の問題を見つける、いちばん安い監視方法**です。

### 5.3.4 SFT

事前学習が終わったら、**同じ学習ループのまま Dataset だけ差し替えて** SFT します。

```python
config = ModelConfig(dim=1024, n_layers=18)          # 5.3.3 と同じ構成にすること
model = Transformer(config)
model.load_state_dict(torch.load("pretrain.pth"))    # 5.3.3 の最後に保存した重みをロード
loader = DataLoader(
    SFTDataset("sft_data.jsonl", tokenizer, max_length=512),  # Dataset を差し替え（5.3.2）
    batch_size=8, shuffle=True,
)
# 学習ループは 5.3.3 と同一。学習率だけ小さく（例：1e-5 〜 5e-5）
```

**これだけです。** 第4章で説明した通り、Pretrain と SFT の違いは
**データと損失マスクだけ**であることが、コードでも確認できます。

---

## 5.4 テキスト生成（推論）

学習したモデルで実際に文章を生成します。

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=0.8, top_k=50):
    for _ in range(max_new_tokens):
        # コンテキスト長を超えたら後ろだけ使う
        idx_cond = idx if idx.size(1) <= model.args.max_seq_len \
                       else idx[:, -model.args.max_seq_len:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature     # ① 温度で割る

        # ② Top-k サンプリング：上位k個以外を除外
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)   # ③ 確率に従って抽選
        idx = torch.cat((idx, idx_next), dim=1)              # ④ 追加してループ
    return idx
```

### サンプリングのパラメータ

| パラメータ | 効果 |
|-----------|------|
| **temperature** | 小さい（0.1）＝ 確定的・堅実。大きい（1.5）＝ 多様・創造的。0 に近づけると常に最頻トークン |
| **top_k** | 確率上位 k 個からのみ選ぶ。変な単語が出るのを防ぐ |
| **top_p（Nucleus）** | 累積確率が p になるまでの候補から選ぶ。より柔軟 |
| **repetition_penalty** | 同じ単語の繰り返しにペナルティ |

#### ★ 同じ1つの分布に、4つのパラメータを順番に掛けてみる

言葉だけでは違いが分からないので、**「公園に」の次の候補8個**を実際に用意して、
それぞれのパラメータを掛けた結果を全部出します（すべて実際に計算した値です）。

```
   logits（モデルの生の出力）
     行き 4.0 ／ 散歩 3.2 ／ 遊び 2.8 ／ 出かけ 2.5 ／ 走り 2.0 ／ 、 1.5 ／ 。 1.2 ／ バナナ −1.0
```

| | 行き | 散歩 | 遊び | 出かけ | 走り | 、 | 。 | バナナ |
|---|---|---|---|---|---|---|---|---|
| **① そのまま**（T=1.0） | 44.3% | 19.9% | 13.3% | 9.9% | 6.0% | 3.6% | 2.7% | 0.3% |
| **② T = 0.5**（尖らせる） | **72.9%** | 14.7% | 6.6% | 3.6% | 1.3% | 0.5% | 0.3% | 0.0% |
| **② T = 1.5**（平坦にする） | 32.8% | 19.3% | 14.7% | 12.1% | 8.7% | 6.2% | 5.1% | **1.2%** |
| **③ top_k = 3** | 57.1% | 25.7% | 17.2% | **0** | **0** | **0** | **0** | **0** |
| **④ top_p = 0.9** | 47.4% | 21.3% | 14.3% | 10.6% | 6.4% | **0** | **0** | **0** |
| **⑤ repetition_penalty = 1.2**<br/>（「行き」を既に使った場合） | **29.0%** | 25.4% | 17.0% | 12.6% | 7.6% | 4.6% | 3.4% | 0.4% |

**1行ずつ、何が起きたのかを読みます。**

| | 仕組み | この表から読めること |
|---|---|---|
| **② temperature** | softmax の前に logits を T で割る | **順位は絶対に変わりません**。1位と2位の差が広がる／縮まるだけ。T=1.5 では「バナナ」が 0.3%→1.2% と**4倍**になっている点に注意（＝おかしな出力が出る余地が増える） |
| **③ top_k** | 上位 k 個だけ残して、残りを −∞ にしてから再度 softmax | 3個に絞ったので、残った3個の確率が**足して100%になるよう再配分**されます（44.3%→57.1%）。欠点は **k が固定**なこと——候補が本当に2個しかない場面でも3個目を拾ってしまう |
| **④ top_p** | 確率の高い順に足していき、**累積が p を超えたところで打ち切る** | 累積は `44.3 → 64.2 → 77.5 → 87.4 → 93.4%`。0.9 を超えるのは5個目なので、**「走り」までが採用**されます（超えた1個は残すのが慣例）。候補の数が**その場の分布に応じて自動で変わる**のが top_k との違い |
| **⑤ repetition_penalty** | すでに出たトークンの logit を、正なら割り、負なら掛ける | 「行き」だけが 44.3%→29.0% に下がり、他が押し上げられます。**1.0 が無効、1.1〜1.2 が実用範囲**。大きすぎると必要な語（助詞や固有名詞）まで避けるようになります |

> 🔑 **top_k と top_p は「切り捨て」、temperature は「傾きの調整」**——別の仕事です。
> だから実務では **top_p ＋ temperature** を組み合わせます
> （まず明らかにおかしい候補を切り、残りの中での大胆さを温度で決める）。
>
> 📝 **順番も決まっています**：`repetition_penalty` → `temperature` → `top_k` → `top_p` → サンプリング。
> 切り捨ててから温度を掛けても、切り捨てた候補は戻ってきません。

> ⚠️ **「バナナ」の行を見てください。** 生の分布ですでに 0.3% あります。
> **1000トークン生成すれば、確率 0.3% の事故は数回起きます。**
> 長い生成でときどき突然おかしくなるのは、この積み重ねです。
> top_k / top_p が「切り捨て」である理由——**低確率の裾を物理的に消す**ためです。

**用途ごとの設定の目安：**

| やりたいこと | temperature | top_p | 備考 |
|---|---|---|---|
| 事実の抽出・分類・JSON 出力 | **0**（または 0.01） | — | 毎回同じ答えが欲しいので、くじ引きをしない |
| コード生成 | 0.2 〜 0.4 | 0.95 | 少しの揺れは許すが、構文は壊したくない |
| 一般的な対話 | 0.7 〜 1.0 | 0.9 〜 0.95 | 既定値がだいたいここ |
| 創作・ブレスト | 1.0 〜 1.3 | 0.95 〜 1.0 | 破綻も許容して幅を取る |

---

## 実測：この構成は本当に学習できるのか

`code/05_llama.py --train` の実際の実行結果です（CPU、約3.5分）。

```
  デバイス: cpu / パラメータ: 3,296,512
  語彙: 346 / 学習トークン: 156,868

    step    train      val      経過
       0   6.0665   5.6821     0.3s
     300   0.6821   0.6980    47.8s
     900   0.6810   0.6793   140.3s
    1199   0.6953   0.6850   186.5s

  生成例:  山に狐がいる。 / 馬が豆を食べる。 / 花子は学校へ行く。
  文法正答率: 231/239 = 96.7%  [合格]
```

確認できること：

1. **初期 loss 6.07 ≈ ln(346) = 5.85** から始まっている（実装が正しい証拠。[00.5章](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/005-deep-learning-basics)）
2. **理論下限（約0.68）まで下がって止まる**（[02.7章](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/027-mini-gpt) と同じ現象）
3. 正答率は [02.7章](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/027-mini-gpt) の GPT-2 構成（98.3%）と**ほぼ同じ**

> 🔑 **小規模では新旧アーキテクチャの性能差は出ません。**
> RMSNorm・RoPE・GQA・SwiGLU の恩恵が現れるのは
> **「長文」「大規模」「推論コスト」**の3点であり、それこそが導入された理由です。
> ちなみにこの生成は、**学習文脈長 64 を超えて 300 トークン**書かせています。
> 04 の学習型位置埋め込みでは、64 より先の位置ベクトルが存在しないので**原理的に不可能**でした。
> RoPE なら回転角を計算し続けるだけなので、少なくとも**動きます**。
>
> ⚠️ ただし「**動いた**」を「**長文に外挿できる**」と読まないでください。
> このコーパスは1文が7〜8トークンで完結し、**遠くの位置を参照する必要が一切ありません**。
> だから遠距離の回転角が壊れていても影響が出ないだけです。
> 本物の長文（前の章を参照する、など）では、5.1.3 の ⚠️ で書いた細工（YaRN など）が必要になります。
>
> 🔑 **教訓**：小さい実験で「動いた」ことは、スケールしたときの保証になりません。
> **どういう条件で動いたのか**を毎回言えるようにしてください。

---

## この章のまとめ

- **LLaMA2 = Transformer Decoder + RMSNorm + RoPE + GQA + SwiGLU**
- 各技術の目的：

| 技術 | 置き換え元 | 目的 |
|------|-----------|------|
| RMSNorm | LayerNorm | 高速化 |
| RoPE | 絶対位置エンコーディング | 相対位置・後から文脈長を伸ばせる（→ 5.1.3 の ⚠️） |
| GQA | MHA | KVキャッシュの**保存量**削減（計算量は減らない → 5.1.4） |
| SwiGLU | ReLU FFN | 表現力向上 |
| Pre-Norm | Post-Norm | 学習の安定化 |
| Weight Tying | — | パラメータ削減 |

- トークナイザは **BPE** で訓練する。特殊トークンと Chat Template の設定が重要
- **Pretrain と SFT のコードはほぼ同じ**。違いは Dataset（損失マスク）と学習率だけ
- 学習の安定化テクニック：ウォームアップ、コサイン減衰、勾配累積、混合精度、勾配クリッピング
- 生成時は temperature / top-k / top-p で「堅実さ ⇔ 創造性」を調整する

## 理解度チェック

1. RMSNorm が LayerNorm より速い理由は何ですか？
2. RoPE が「相対位置を扱える」と言われるのはなぜですか？
3. GQA は何を減らすための技術ですか？
4. `Y = chunk[1:]` としているのはなぜですか？
5. 勾配累積を使うと何ができるようになりますか？
6. temperature を 0 に近づけると出力はどうなりますか？
7. SwiGLU で `hidden_dim` を 2/3 に縮めているのはなぜですか？
8. Weight Tying（重み共有）が成立するのはなぜですか？

<details>
<summary><b>▶ 解答を見る</b></summary>

1. **平均を引く処理を省いている**からです。LayerNorm は平均と分散の両方を計算しますが、
   RMSNorm は二乗平均平方根（RMS）だけで割ります。
   研究の結果、LayerNorm の効果の大半は「平均を0にすること」ではなく
   「**スケールを揃えること**」だと分かったため、平均の計算を省いても性能はほぼ変わりません。
   これで 7〜64% 高速化します。
2. **Q と K の内積を取ると、回転角の差（＝位置の差）だけが残る**ためです。
   RoPE は位置 `m` のベクトルを角度 `mθ` だけ回転させますが、
   位置 `m` の Q と位置 `n` の K の内積を計算すると、結果は `(m − n)` にのみ依存する式になります。
   **絶対位置を使って回転させているのに、内積の結果は相対位置で決まる**——ここが美しい点です。
   ⚠️ ただし「だから長文にそのまま外挿できる」わけではありません（5.1.3 の ⚠️ 参照）。
   素の RoPE は学習長を超えると崩れるので、PI / YaRN のような細工が必要です。
3. **推論時の KV キャッシュのメモリ**です。
   Q は全ヘッド分持ちますが、K と V は少ないヘッド数しか持たず、複数の Q ヘッドで共有します。
   コード上は `wk`, `wv` の出力次元だけが `n_kv_heads * head_dim` と小さくなっている点に対応します。
4. **「次トークン予測」を実装するため**です。`X = chunk[:-1]`、`Y = chunk[1:]` とすることで、
   `Y` は `X` を**1つ左にずらしたもの**になります。
   つまり各位置で「その次に来るトークン」が正解ラベルになります。これが CLM の実装の全てです。
5. **GPU メモリを増やさずに、実質的なバッチサイズを大きくできます。**
   通常は毎回 `optimizer.step()` しますが、勾配累積では数回分の勾配を溜めてから1回だけ更新します。
   `batch_size=4` × `ACCUM_STEPS=8` なら、実質バッチサイズ 32 で学習したのと同じ効果になります。
   （仕組みとしては `zero_grad()` を毎回呼ばないだけです → [00.5章](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/005-deep-learning-basics)）
6. **完全に決定的になり、毎回いちばん確率の高いトークンだけを選ぶようになります**（貪欲法 / greedy）。
   temperature は softmax の直前で logits を割る値なので、
   小さくすると logits の差が拡大され、最大値に確率が集中します。
   再現性は得られますが、**同じ文を延々と繰り返しがち**という副作用があります。
7. **線形層が1つ増えた分のパラメータ数を相殺するため**です。
   通常の FFN は `w1`, `w2` の2層ですが、SwiGLU はゲート用の `w3` が加わって3層になります。
   そのまま `hidden_dim = 4 × dim` にするとパラメータが1.5倍になってしまうので、
   `2/3` を掛けて元の FFN とほぼ同じパラメータ数に揃えています。
   （その上で `multiple_of` の倍数に切り上げるのは GPU の演算効率のためです）
8. **入力 Embedding（語彙数 × 次元）と出力層（次元 × 語彙数）が、形として転置の関係にある**からです。
   どちらも「トークンとベクトルの対応表」を表しているので、同じ行列を使い回せます。
   語彙数 × 次元のパラメータを丸ごと1つ節約でき、性能も落ちません。
   小さいモデルほど Embedding の占める割合が大きいので、効果が大きくなります。

</details>

---

**次へ** → [06. 学習フローの実践](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/06-training-pipeline)
／ 2026年の実物との差分を埋めるなら → **[05.5. MoE と現代アーキテクチャ](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/055-moe-modern-arch)**
