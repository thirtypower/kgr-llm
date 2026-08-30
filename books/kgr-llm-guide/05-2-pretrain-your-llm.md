---
title: "05. 自分で LLM を作る（後編）トークナイザ訓練と事前学習"
---

> [前編](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/05-build-your-own-llm) の続きです。
> 前編で書いた LLaMA2 のモデル本体に、**トークナイザを訓練して与え、実際に事前学習を回し、文章を生成させる**までを扱います。

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
