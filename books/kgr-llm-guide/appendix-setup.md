---
title: "環境構築ガイド（コードを動かすための完全手順）"
---

> **対象**：Python を入れるところから始める人。
> ここに書いてある通りに進めれば、[`code/`](https://github.com/thirtypower/kgr-llm/tree/main/code) の全スクリプトが動きます。
> 所要時間：**初回セットアップ 10〜20分**（うち大半は PyTorch のダウンロード待ち）。

---

## 0. 必要なもの

| 項目 | 要件 |
|------|------|
| OS | Windows / macOS / Linux どれでも可 |
| Python | **3.9 以上**（この教材は 3.11 で動作確認済み） |
| メモリ | 4GB 以上あれば十分 |
| GPU | **不要**（基本コース）。例外は `06_pretrain.py --full` と `07_lora_sft.py` の2つだけで、これらは GPU または Colab 推奨（→ [8節](#8-章ごとの追加ライブラリと-gpu-要件の早見表)） |
| ディスク | 約 3GB（PyTorch 本体が大きい） |
| ネット接続 | 初回の `pip install` のときだけ必要 |

---

## 1. Python が入っているか確認する

ターミナル（Windows なら PowerShell、macOS なら「ターミナル」）を開いて：

```bash
python --version
```

`Python 3.11.4` のように **3.9 以上**が表示されれば OK。次の節へ。

**表示されない／2.x が表示される場合：**

| OS | インストール方法 |
|----|----------------|
| Windows | [python.org](https://www.python.org/downloads/) からインストーラを取得。**「Add python.exe to PATH」に必ずチェック**を入れる |
| macOS | `brew install python` （Homebrew が無ければ python.org のインストーラでも可） |
| Ubuntu/Debian | `sudo apt install python3 python3-pip python3-venv` |

> 💡 環境によっては `python` ではなく `python3` というコマンド名です。
> 以降のコマンドで `python` が見つからないと言われたら `python3` に読み替えてください。

---

## 2. 仮想環境を作る（推奨）

**仮想環境**とは、このプロジェクト専用の「隔離された Python 環境」です。
システム全体を汚さず、失敗したらフォルダごと消してやり直せます。

```bash
# このリポジトリのフォルダに移動してから
cd kgr-llm

# ① 仮想環境を作る（.venv という名前のフォルダができる）
python -m venv .venv

# ② 有効化する（★ ターミナルを開き直すたびに必要）
#    Windows (PowerShell):
.venv\Scripts\Activate.ps1
#    Windows (コマンドプロンプト):
.venv\Scripts\activate.bat
#    macOS / Linux:
source .venv/bin/activate
```

有効化に成功すると、プロンプトの先頭に `(.venv)` と表示されます。

> ⚠️ **PowerShell で「スクリプトの実行が無効」というエラーが出た場合**：
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
> を実行してから、もう一度 Activate してください。

---

## 3. PyTorch をインストールする

この教材で必要なライブラリは **PyTorch ただ1つ**です。

```bash
pip install torch
```

- ダウンロードは数百MB〜2GB あります。**数分かかるのが正常**です
- この教材は CPU しか使わないので、CUDA 対応版かどうかは気にしなくて構いません

**確認：**

```bash
python -c "import torch; print(torch.__version__)"
```

`2.x.x` のようにバージョンが表示されれば完了です。

> 💡 `CUDA available: False` と出ても**全く問題ありません**。
> この教材のコア教材（`01_`〜`05_` と `06_pretrain.py --demo`）は
> すべて CPU で動くように設計されています。GPU が要るのは
> `06_pretrain.py --full`（本物のデータで215M学習）と `07_lora_sft.py`（LoRA）だけで、
> どちらも Colab という逃げ道があります（→ 8節の早見表）。

---

## 4. 動かす（実行順と所要時間）

`code/` フォルダに移動して、番号順に実行します。

```bash
cd code
```

| 順 | コマンド | 実測時間※ | 何が起きるか |
|----|---------|----------|-------------|
| 1 | `python 01_tensor_autograd.py` | **約4秒** | 勾配・損失・Optimizer を最小の例で観察 |
| 2 | `python 02_attention_step_by_step.py` | **約2秒** | Attention の途中の行列を全部表示 |
| 3 | `python 03_bpe_tokenizer.py` | **1秒未満** | BPE トークナイザの訓練を1マージずつ観察 |
| 4 | `python 04_minigpt.py --sanity` | **約7秒** | GPT 実装の自己検査6項目だけ実行 |
| 5 | `python 04_minigpt.py` | **約4分** | GPT を実際に学習 → 生成 → 文法採点 |
| 6 | `python 05_llama.py` | **約11秒** | LLaMA2 の5つの技術を数値で検証 |
| 7 | `python 05_llama.py --train` | **約3.5分** | LLaMA2 構成で学習 → 生成 → 採点 |
| 8 | `python 06_pretrain.py --demo` | **約3分** | 事前学習の全工程（データ→BPE→学習→保存→生成）を1周 |
| 9 | `python 08_moe.py` | **約10秒** | MoE の仕組みを数値で検証（パラメータ・ルータ・KVキャッシュ） |
| 10 | `python 08_moe.py --train` | **約2.5分** | ルータが偏る（＝専門家が遊ぶ）現象を実測 |
| 11 | `python 09_inference.py` | **約20秒** | KV キャッシュ・文脈長・バッチの効果を実測 |
| 12 | `python 09_inference.py --quant` | **約1分** | 量子化で品質がどこで崩れるかを実測 |

※ クラウドの一般的な CPU（GPU なし）での実測値。手元のマシンでは前後します。

> 💡 上の12本はすべて **PyTorch だけ・CPU だけ**で動きます。
> `06_pretrain.py --full`（215M の本格事前学習）と `07_lora_sft.py`（LoRA 微調整）だけは
> 追加ライブラリと GPU（または Colab）が必要です → [8節](#8-章ごとの追加ライブラリと-gpu-要件の早見表)

### 成功したかどうかの判定

**手順5（`04_minigpt.py`）が最重要です。** 最後にこう表示されれば成功：

```
  生成した文: 420 文中 413 文が文法的に正しい  →  正答率 98.3%
  今回の結果: 合格
```

この時点であなたは「**LLM を自分の手でゼロから学習させた**」ことになります。

---

## 5. トラブルシューティング

| 症状 | 原因と対処 |
|------|-----------|
| `ModuleNotFoundError: No module named 'torch'` | PyTorch が入っていない、または**仮想環境を有効化し忘れている**。プロンプトに `(.venv)` があるか確認 → 無ければ手順2の②を再実行 |
| `pip install torch` が異常に遅い / 途中で切れる | ファイルが大きいだけで正常。切れたら同じコマンドを再実行（途中から再開されます） |
| 社内ネットワークで `pip` がプロキシエラーになる | `pip install torch --proxy http://プロキシ:ポート`。それでもダメなら情報システム部門にプロキシ設定を確認 |
| Windows で日本語が文字化けする | 各スクリプトが自動対処しますが、それでも化けるなら実行前に `chcp 65001`、または `python -X utf8 01_tensor_autograd.py` |
| `python` と打つと Microsoft Store が開く（Windows） | 「設定 → アプリ → アプリ実行エイリアス」で `python.exe` の Store 版を**オフ**にする。または `py` コマンドを使う |
| 学習が表の時間より大幅に遅い | 古い CPU では2〜3倍かかることがあります。`python 04_minigpt.py --steps 500` でステップ数を減らしても学習の様子は確認できます（正答率は下がります） |
| `AssertionError` で止まる | **バグ検出が正しく働いた**ということです。エラーメッセージに日本語で原因が書いてあります。改造した場合は元に戻して再確認を |
| メモリ不足で落ちる | このコード規模では稀ですが、他のアプリを閉じる。それでもダメなら `04_minigpt.py` の `--batch-size 16` を試す |

---

## 6. Google Colab で動かす（自分の PC に入れたくない場合）

ブラウザだけで試せます。

1. [colab.research.google.com](https://colab.research.google.com/) で新規ノートブックを作成
2. セルに以下を貼り付けて実行：

```python
!git clone https://github.com/thirtypower/KGR-LLM.git
%cd KGR-LLM/kgr-llm/code
!python 04_minigpt.py
```

Colab には PyTorch が最初から入っているので、インストール不要です。

> ⚠️ リポジトリが非公開の場合、`git clone` は失敗します。
> その場合は Colab 左側のフォルダアイコンから `code/` フォルダの `.py` ファイルを
> 全部アップロードして、`!python 04_minigpt.py` を実行してください
> （`_bpe.py` `_corpus.py` `_console.py` の3つを忘れずに）。
> なお `09_inference.py` は `04_minigpt.py` を読み込むので、そちらも一緒に置いてください。

---

## 7. 6章以降（Hugging Face エコシステム）に進むとき

基本コース（PyTorch のみ）を終えて、実際に大きめのモデルを扱う段階になったら、
次の 8 節の早見表に従って必要なものだけ追加インストールしてください。

---

## 8. 章ごとの追加ライブラリと GPU 要件の早見表

### 追加ライブラリ

| どこまでやるか | 追加で必要なもの | 備考 |
|--------------|----------------|------|
| 〜5章の写経・`code/` の全デモ（`08_` `09_` を含む） | **なし**（PyTorch のみ） | |
| 5章フル学習（`06_pretrain.py --full`） | `pip install datasets tokenizers` | Wikipedia のダウンロードと BPE 訓練に使用 |
| 6章（SFT / LoRA / DPO、`07_lora_sft.py`） | `pip install transformers datasets peft trl accelerate` | |
| 6章の DeepSpeed（ZeRO） | `pip install deepspeed` | ⚠ **Windows 非対応**。WSL2 か Linux で。そもそも複数 GPU が前提なので **Colab / クラウド推奨** |
| 7章の評価 | `pip install lm-eval` | |
| 7章の RAG | `pip install openai sentence-transformers` | |
| 7章の Agent | `pip install openai` ＋ サーバ側に vLLM（Linux/WSL2）または Ollama（Windows 可、手軽） | |

### GPU 要件

| やること | GPU | 備考 |
|---------|-----|------|
| `code/` の番号付き全スクリプト（01〜05, 06 `--demo`, 08, 09） | **不要（CPU のみ）** | 数秒〜数分 |
| `06_pretrain.py --full`（215M 事前学習） | **VRAM 12GB〜**（RTX 3060 12GB 可） | 5万記事で一晩程度。本格的にやるほど時間はデータ量に比例 |
| `07_lora_sft.py`（0.5B の LoRA SFT） | **VRAM 6GB〜 または Colab 無料 T4** | 500件なら T4 で十数分 |
| 7B クラスの LoRA / QLoRA | VRAM 12〜24GB | QLoRA なら 12GB でも可 |
| 7B クラスの推論（vLLM で Agent 用サーバ等） | **VRAM 16GB〜** | 足りなければ 1.5B に落とすか Ollama の量子化モデルで |

> 💡 **GPU が無い場合の結論**：学習系は Google Colab（無料 T4）、
> 推論系は Ollama（CPU でも量子化モデルなら動く）が現実的な選択肢です。

---

**次へ** → 準備ができたら [README の学習コース](https://github.com/thirtypower/kgr-llm#4-おすすめの学習の進め方) から自分のコースを選んでください。
