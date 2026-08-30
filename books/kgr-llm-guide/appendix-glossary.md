---
title: "用語集（日本語 ⇔ 英語）"
---

> 分からない用語が出てきたらここに戻ってください。
> **英語のスペルも一緒に覚える**と、論文やライブラリのドキュメントが読めるようになります。

### この用語集の構成

| 節 | 内容 | 主に対応する章 |
|----|------|--------------|
| **@** | 深層学習の基礎（テンソル・勾配・Optimizer） | 00.5 |
| **A** | 基礎・NLP（トークン化・埋め込み） | 01 / 02.6 |
| **B** | Transformer とアーキテクチャ | 02 / 05 |
| **C** | モデルの種類（BERT / GPT 系） | 03 |
| **D** | 学習（Pretrain / SFT / LoRA / 分散） | 04 / 05 / 06 |
| **E** | 能力・現象（創発・ICL・ハルシネーション） | 04 |
| **F** | 推論・生成（temperature / サンプリング） | 05 / 02.7 |
| **G** | 応用（RAG / Agent / 評価） | 07 |
| **H** | 強化学習（GRPO / RLVR） | 08 |
| **I** | 数値の読み方（7B / bf16 / メモリ概算） | 全章 |
| **J** | 現代アーキテクチャ（MoE / MLA / 長文化） | 05.5 |
| **K** | 推論の最適化（KVキャッシュ / 量子化 / 配信） | 07.5 |

---

## @. 深層学習の基礎

> 対応章：[00.5. 深層学習の基礎](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/005-deep-learning-basics)

| 日本語 | 英語 | 意味 |
|--------|------|------|
| テンソル | Tensor | 多次元の数値の箱。0次元=スカラー、1次元=ベクトル、2次元=行列 |
| 形 | Shape | テンソルの各次元の大きさ。`(batch, seq_len, dim)` など |
| 層 | Layer | `y = W x + b` 1枚ぶんの計算。これを重ねることが「深層」の「深」 |
| 全結合層 / 線形層 | Fully Connected / Linear Layer | 入力の全成分を使って出力の各成分を作る層。`nn.Linear` |
| パラメータ | Parameter | 層の中の `W` と `b` の数値そのもの。「7B」＝これが70億個 |
| 活性化関数 | Activation Function | 層と層の間に挟む非線形変換。無いと何枚積んでも1枚に潰れる |
| 正則化 | Regularization | 学習データへの当てはまりを少し犠牲にして汎化性能を上げる工夫（weight decay / Dropout など） |
| 重み減衰 | Weight Decay | 更新のたびに重みを 0 方向へ少し引き戻す正則化。AdamW の "W" |
| 順伝播 | Forward (Pass) | 入力から出力（予測）を計算すること |
| 勾配 | Gradient | 「そのパラメータを増やしたら損失がどう変わるか」を表す値 |
| 自動微分 | Autograd / Automatic Differentiation | 計算を記録し、微分を自動で計算する仕組み |
| 計算グラフ | Computational Graph | 「どんな演算をしたか」の記録。逆伝播はこれを逆にたどる |
| 誤差逆伝播 | Backpropagation | 計算グラフを逆順にたどり、全パラメータの勾配を1パスで求める |
| 連鎖律 | Chain Rule | 合成関数の微分則。逆伝播の数学的な土台 |
| 内積 | Dot Product | 同じ位置の数字を掛けて全部足す。「向きの近さ」を1つの数字にする |
| softmax | Softmax | 生の点数を「合計1の確率」に変換する（全部 exp して合計で割る） |
| ロジット | Logits | softmax に入れる前の生のスコア。正負も大小も自由 |
| 最適化手法 | Optimizer | 勾配を使って重みをどう更新するかを決める仕組み |
| SGD | Stochastic Gradient Descent | 最も素朴な更新則 `w ← w − lr·g` |
| AdamW | Adam with decoupled Weight decay | パラメータごとに歩幅を自動調整。**LLM の事実上の標準** |
| パープレキシティ | Perplexity (PPL) | `exp(loss)`。「実質何択で迷っているか」を表す指標 |
| 検証データ | Validation Set | 学習に使わず、汎化性能の確認に使うデータ |
| 初期損失の検査 | — | 学習開始時の loss が `ln(語彙数)` かを見るバグ検出法（[00.5章](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/005-deep-learning-basics)） |

---

## A. 基礎・NLP

| 日本語 | 英語 | 意味 |
|--------|------|------|
| 自然言語処理 | Natural Language Processing (NLP) | 人間の言葉をコンピュータで扱う分野 |
| トークン | Token | モデルが扱う言葉の最小単位 |
| トークン化 | Tokenization | 文をトークンに分割すること |
| サブワード分割 | Subword Tokenization | 単語をさらに細かい部品に割る方式 |
| BPE | Byte Pair Encoding | 頻出ペアを結合していくトークン化手法 |
| バイトレベル BPE | Byte-Level BPE | 256個の全バイトから始める BPE。**未知語が原理的に発生しない** |
| マージ規則 | Merge Rules | BPE の訓練成果。「どのペアを結合するか」の順序つきリスト |
| 事前分割 | Pre-tokenization | BPE を掛ける前に文をざっくり塊に切る処理 |
| 語彙 | Vocabulary | モデルが知っているトークンの一覧 |
| 未知語 | OOV (Out-Of-Vocabulary) | 語彙に無い単語 |
| 圧縮率 | Compression Rate | 1トークンあたり何文字を表せるか。言語によって大きく違う |
| 埋め込み | Embedding | 言葉を意味を持つ数値ベクトルにしたもの |
| 単語ベクトル | Word Vector | 同上（特に単語単位のもの） |
| コーパス | Corpus | 学習に使う大量のテキストデータ |
| 形態素解析 | Morphological Analysis | 日本語を単語に分け品詞を振る処理 |
| 固有表現抽出 | Named Entity Recognition (NER) | 人名・地名などを抜き出すタスク |
| 品詞タグ付け | POS Tagging | 各単語に品詞を付けるタスク |
| 言語モデル | Language Model (LM) | 次の単語の確率を予測するモデル |
| N-gram | N-gram | 直前 N-1 語から次を予測する統計手法 |

---

## B. Transformer とアーキテクチャ

| 日本語 | 英語 | 意味 |
|--------|------|------|
| 注意機構 | Attention | どの単語に注目すべきかを計算する仕組み |
| 自己注意 | Self-Attention | Q・K・V を同じ入力から作る Attention |
| マスク付き自己注意 | Masked / Causal Self-Attention | 未来のトークンを見せない Attention |
| 多頭注意 | Multi-Head Attention (MHA) | 複数の視点で並列に Attention を行う |
| グループ化クエリ注意 | Grouped-Query Attention (GQA) | K/V を共有して KVキャッシュを削減 |
| クエリ / キー / バリュー | Query / Key / Value | Attention の3要素 |
| 交差注意 | Cross-Attention | Decoder が Encoder の出力を参照する |
| 順伝播ネットワーク | Feed-Forward Network (FFN) | Attention の後段の2〜3層の全結合層 |
| 層正規化 | Layer Normalization | 各サンプル内で数値を正規化する |
| RMSNorm | Root Mean Square Norm | 平均計算を省いた高速な正規化 |
| 残差接続 | Residual / Skip Connection | 入力をそのまま出力に足す |
| 位置エンコーディング | Positional Encoding | トークンの順序情報を与える |
| 回転位置埋め込み | RoPE (Rotary Position Embedding) | 回転で位置を表す。相対位置に強い |
| 活性化関数 | Activation Function | ReLU / GELU / SiLU など非線形変換 |
| SwiGLU | SwiGLU | ゲート機構つきの FFN。LLaMA が採用 |
| Flash Attention | Flash Attention | メモリ効率の良い Attention 実装 |
| KVキャッシュ | KV Cache | 生成時に過去の K/V を再利用する仕組み |
| コンテキスト長 | Context Length / Window | 一度に扱えるトークン数 |
| 重み共有 | Weight Tying | 入力 Embedding と出力層で同じ重みを使う |
| QK-Norm | QK-Norm | Q と K を正規化してから内積を取る。大規模学習の安定化 |
| Attention sink | Attention Sink | モデルが文頭トークンに大量の注意を割く現象。「注意の捨て場」として働く |

---

## C. モデルの種類

| 日本語 | 英語 | 意味 |
|--------|------|------|
| 事前学習言語モデル | Pre-trained Language Model (PLM) | 大量データで事前学習されたモデル |
| 大規模言語モデル | Large Language Model (LLM) | 数百億以上のパラメータを持つ言語モデル |
| エンコーダのみ | Encoder-only | BERT 系。理解タスクに強い |
| デコーダのみ | Decoder-only | GPT 系。生成タスクに強い。現代 LLM の本流 |
| マスク言語モデル | Masked Language Model (MLM) | 穴埋めで学習（BERT） |
| 因果言語モデル | Causal Language Model (CLM) | 次単語予測で学習（GPT） |
| Base モデル | Base Model | 事前学習のみのモデル。指示に従わない |
| Instruct / Chat モデル | Instruct / Chat Model | SFT 済みで指示に従うモデル |
| マルチモーダル | Multimodal | テキスト以外（画像・音声）も扱える |

---

## D. 学習

| 日本語 | 英語 | 意味 |
|--------|------|------|
| 事前学習 | Pretraining | 大量テキストで基礎能力を獲得する段階 |
| ファインチューニング | Fine-tuning | 学習済みモデルを特定用途に調整する |
| 教師ありファインチューニング | SFT (Supervised Fine-Tuning) | 指示-応答ペアで学習する段階 |
| 人間フィードバック強化学習 | RLHF | 人間の好みを強化学習で反映 |
| 直接選好最適化 | DPO (Direct Preference Optimization) | 報酬モデル不要の選好学習 |
| 選好アライメント | Preference Alignment | 人間の好みに合わせる工程全般 |
| 効率的パラメータ微調整 | PEFT | 少数パラメータだけ学習する手法群 |
| LoRA | Low-Rank Adaptation | ΔW を低ランク行列で近似する PEFT |
| QLoRA | Quantized LoRA | 4bit 量子化 + LoRA |
| ランク | Rank (r) | LoRA の行列の細さ。小さいほど軽い |
| 量子化 | Quantization | 数値精度を落としてメモリ削減 |
| 損失関数 | Loss Function | モデルの誤差を測る関数 |
| 交差エントロピー | Cross Entropy | 分類・言語モデルの標準的な損失 |
| 損失マスク | Loss Mask | 一部トークンを損失計算から除外する |
| 勾配降下法 | Gradient Descent | 損失を減らす方向に重みを更新する |
| 勾配累積 | Gradient Accumulation | 複数バッチ分の勾配を溜めてから更新 |
| 勾配チェックポイント | Gradient Checkpointing | 中間値を捨てて再計算しメモリ節約 |
| 勾配クリッピング | Gradient Clipping | 勾配の大きさに上限を設ける |
| 混合精度学習 | Mixed Precision (AMP) | bf16/fp16 を使って高速・省メモリ化 |
| ウォームアップ | Warmup | 学習率を徐々に上げる期間 |
| エポック | Epoch | データ全体を1周すること |
| バッチサイズ | Batch Size | 一度に処理するサンプル数 |
| 学習率 | Learning Rate | 1回の更新でどれだけ重みを動かすか |
| 過学習 | Overfitting | 学習データに適合しすぎて汎化しない |
| スケーリング則 | Scaling Laws | 規模と性能の関係を表す経験則 |
| 分散学習 | Distributed Training | 複数 GPU で学習を分担する |
| ZeRO | Zero Redundancy Optimizer | DeepSpeed のメモリ分割技術 |
| 知識蒸留 | Knowledge Distillation | 大きいモデルの知識を小さいモデルへ移す |
| SimPO | Simple Preference Optimization | 参照モデル不要・**長さで正規化**した DPO の改良（→ 06章 6.4.2） |
| 長さバイアス | Length Bias | 選好学習で出力が冗長になる偏り。DPO の代表的な弱点 |

---

## E. 能力・現象

| 日本語 | 英語 | 意味 |
|--------|------|------|
| 創発能力 | Emergent Ability | 規模が閾値を超えると突然現れる能力 |
| 文脈内学習 | In-Context Learning (ICL) | プロンプト内の例だけでタスクを学ぶ |
| ゼロショット | Zero-shot | 例なしでタスクをこなす |
| フューショット | Few-shot | 数個の例を与えてタスクをこなす |
| 指示追従 | Instruction Following | 自然言語の指示を理解して実行する |
| 思考の連鎖 | Chain-of-Thought (CoT) | 途中の考えを言語化して段階的に解く |
| ハルシネーション | Hallucination | もっともらしい嘘を生成する現象 |
| 報酬ハッキング | Reward Hacking | 報酬の抜け穴を突く望ましくない挙動 |
| データ汚染 | Data Contamination | 評価問題が学習データに混入すること |

---

## F. 推論・生成

| 日本語 | 英語 | 意味 |
|--------|------|------|
| 推論 | Inference | 学習済みモデルで出力を生成すること |
| 自己回帰生成 | Autoregressive Generation | 1トークンずつ順に生成する方式 |
| 温度 | Temperature | 出力のランダム性を調整する値 |
| Top-k サンプリング | Top-k Sampling | 上位 k 個の候補から抽選する |
| Top-p サンプリング | Top-p / Nucleus Sampling | 累積確率 p までの候補から抽選する |
| ビームサーチ | Beam Search | 複数候補を並行して探索する生成法 |
| プロンプト | Prompt | モデルへの入力文 |
| システムプロンプト | System Prompt | モデルの役割を指定する指示 |
| チャットテンプレート | Chat Template | 会話を特殊トークンで整形する形式 |
| 生成開始プロンプト | add_generation_prompt | 末尾の `<\|im_start\|>assistant\n`。**付け忘れるとモデルが自問自答する** |
| 特殊トークン | Special Token | `<s>` `<\|im_start\|>` など制御用トークン |
| 貪欲法 | Greedy Decoding | 常に最頻トークンを選ぶ生成法（temperature = 0） |
| 詰め込み | Packing | 複数文書を繋げて固定長に切り、パディングの無駄をなくす |

---

## G. 応用

| 日本語 | 英語 | 意味 |
|--------|------|------|
| 検索拡張生成 | RAG (Retrieval-Augmented Generation) | 外部文書を検索して回答の根拠にする |
| チャンク分割 | Chunking | 文書を適切な大きさに切り分ける |
| ベクトルデータベース | Vector Database | ベクトルを保存し類似検索する DB |
| コサイン類似度 | Cosine Similarity | ベクトルの向きの近さ |
| リランキング | Reranking | 検索結果を精査して並べ替える |
| ハイブリッド検索 | Hybrid Search | ベクトル検索とキーワード検索の併用 |
| エージェント | Agent | 自律的にツールを使いタスクを遂行する |
| 関数呼び出し | Function Calling | モデルがツールを JSON で指定する仕組み |
| ReAct | Reasoning + Acting | 思考→行動→観察を繰り返す枠組み |
| ツール | Tool | Agent が使える外部機能 |
| サンドボックス | Sandbox | 隔離された安全な実行環境 |
| ベンチマーク | Benchmark | 性能を測る標準テストセット |
| リーダーボード | Leaderboard | モデルの性能順位表 |

---

## H. 強化学習（第8章）

| 日本語 | 英語 | 意味 |
|--------|------|------|
| 強化学習 | Reinforcement Learning (RL) | 報酬を最大化するよう試行錯誤で学ぶ |
| 方策 | Policy | 状態から行動を選ぶ関数（＝LLM 本体） |
| 報酬 | Reward | 行動の良さを表すスコア |
| 報酬モデル | Reward Model | 出力に点数を付けるモデル |
| 価値モデル | Value Model / Critic | 将来の期待報酬を推定するモデル |
| 優位性 | Advantage | 基準よりどれだけ良かったかの差分 |
| 軌跡 | Trajectory | 生成された一連の出力 |
| ロールアウト | Rollout | モデルに実際に生成させること |
| PPO | Proximal Policy Optimization | 標準的な強化学習アルゴリズム |
| GRPO | Group Relative Policy Optimization | グループ内比較で価値モデルを不要にした手法 |
| 検証可能な報酬 | Verifiable Reward (RLVR) | プログラムで自動判定できる報酬 |
| クリッピング | Clipping | 更新幅を制限して学習を安定させる |
| KLダイバージェンス | KL Divergence | 2つの確率分布の隔たりを測る量 |
| 重要度サンプリング | Importance Sampling | 新旧方策の確率比で補正する手法 |
| On-Policy 蒸留 | On-Policy Distillation | 生徒の軌跡を先生が採点する蒸留 |
| 観測マスク | Observation Masking | 環境が返したトークンを学習対象外にする |
| コールドスタート | Cold Start | RL 前に SFT で基礎能力を仕込むこと |

---

## I. 数値の読み方

| 表記 | 意味 |
|------|------|
| **7B** | 7 Billion = 70億パラメータ |
| **175B** | 1750億パラメータ |
| **1.5T tokens** | 1.5 Trillion = 1兆5000億トークン |
| **fp32 / fp16 / bf16** | 浮動小数点の精度。32bit / 16bit |
| **int8 / int4 (NF4)** | 量子化後の整数精度 |
| **4K / 128K context** | コンテキスト長 4096 / 131072 トークン |
| **d_model = 768** | 隠れ層の次元数 |
| **r = 8** | LoRA のランク |

### パラメータ数からメモリ量を概算する

```
推論時（重みだけ）：
  パラメータ数 × バイト数
  例：7B を bf16（2バイト）→ 約 14GB
      7B を int4（0.5バイト）→ 約 3.5GB

学習時（フル微調整、Adam ＋ 混合精度）：
  重み(2) + 勾配(2) + Adam状態(12) = パラメータ数 × 16バイト
  例：7B → 14GB + 14GB + 84GB = 約 112GB（+ 活性値）

  ※ Adam状態の 12バイト の内訳：
      m（勾配の移動平均）      fp32 = 4
      v（勾配の2乗の移動平均） fp32 = 4
      fp32 マスター重み        fp32 = 4
    「個数は m と v の2つ、バイト数は 2byte 重みの6倍」→ 詳細は 06章 6.1.5

学習時（LoRA）：
  重み(2) + LoRA分のみ ≒ パラメータ数 × 2バイト + α
  例：7B → 約 16〜20GB  ← 24GB の GPU で動く

推論時（KV キャッシュ）：★ 長文ではモデル本体より大きくなる
  2 × 層数 × KVヘッド数 × head_dim × トークン数 × バイト数
  例：LLaMA-2 7B（層32 / GQA 8組 / head_dim 128 / bf16）
        4K   →  0.5GB
       32K   →  4GB
      128K   → 16GB    ← モデル本体 14GB を超える
  ※ これが同時利用者の人数ぶん必要 → 詳細は 07.5章
```

---

## J. 現代アーキテクチャ（第05.5章）

> 対応章：[05.5. MoE と現代アーキテクチャ](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/055-moe-modern-arch)

| 日本語 | 英語 | 意味 |
|--------|------|------|
| 専門家混合 | MoE (Mixture of Experts) | FFN を N 個に増やし、トークンごとに top-k 個だけ使う構造 |
| 総パラメータ / 活性パラメータ | Total / Active Parameters | 「持っている量」と「1トークンで実際に使う量」。MoE では後者が推論費用を決める |
| ルータ / ゲート | Router / Gate | どの専門家を使うかを決める `nn.Linear` 1枚 |
| 負荷分散の補助損失 | Auxiliary Load-Balancing Loss | 専門家の使用を均す罰則。本来の損失と競合するのが弱点 |
| ルータの崩壊 | Routing Collapse | 少数の専門家だけが選ばれ、残りが遊ぶ失敗。**loss では気づけない** |
| 共有専門家 | Shared Expert | 全トークンが必ず通る専門家。共通知識の置き場（DeepSeekMoE 系） |
| 専門家並列 | Expert Parallelism | 専門家を GPU 間に分散する。毎層 all-to-all 通信が発生する |
| MLA | Multi-head Latent Attention | K/V を細い潜在ベクトルに圧縮して保存する。GQA の次 |
| 位置補間 | PI (Position Interpolation) | 位置番号を薄めて文脈長を伸ばす。近距離の解像度が落ちる |
| YaRN / NTK-aware | YaRN / NTK-aware scaling | **周波数ごとに扱いを変えて** RoPE を伸ばす。現在の標準 |
| 中央の見落とし | Lost in the Middle | 長文の**真ん中**に置いた情報を取り落とす現象 |
| スライディングウィンドウ注意 | Sliding Window Attention | 直近 W トークンだけを見る。数層に1層は全部見る層を挟む |
| 線形注意 | Linear Attention | softmax(QKᵀ) をやめ、固定サイズの状態を更新する。計算量が長さに比例 |
| 推論時スケーリング | Test-Time Compute Scaling | 答えるときに長く考えさせて性能を上げる（推論モデル） |

---

## K. 推論の最適化（第07.5章）

> 対応章：[07.5. 推論を速くする](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/075-fast-inference)

| 日本語 | 英語 | 意味 |
|--------|------|------|
| プリフィル | Prefill | 入力プロンプト全体を1回で処理する段階。**計算能力律速** |
| デコード | Decode | 1トークンずつ生成する段階。**メモリ帯域律速** |
| 初トークン遅延 | TTFT (Time To First Token) | 最初の1トークンが出るまでの待ち時間。prefill で決まる |
| 生成速度 | TPS (Tokens Per Second) | 出始めてからの速度。decode で決まる |
| メモリ帯域律速 | Memory-Bound | 演算器は暇で、メモリからの読み出しが渋滞している状態 |
| PagedAttention | PagedAttention | KV キャッシュをブロック単位で貸し出し、断片化をなくす（OS のページング） |
| 連続バッチング | Continuous Batching | 1トークンの区切りでバッチの中身を入れ替える |
| 投機的デコード | Speculative Decoding | 小さい下書きモデルの k トークンを本命モデルが1回で検証。**品質は変わらない** |
| 受理率 | Acceptance Rate | 投機的デコードで下書きが採用される割合。低いと逆に遅くなる |
| プロンプトキャッシュ | Prefix / Prompt Caching | 共通の前置きの KV キャッシュを再利用する。**変わるものは後ろに置く** |
| GPTQ / AWQ | GPTQ / AWQ | 校正データで誤差を補正しながら int4 量子化する手法 |
| GGUF | GGUF | llama.cpp のファイル形式。CPU / Mac 向け量子化の標準 |
| FP8 | FP8 | 8bit 浮動小数点。品質はほぼ無劣化で、新しい GPU がハード対応 |
| 飽和 | Saturation | ベンチマークで上位モデルが差を付けられなくなった状態（→ 07章 7.1.1） |

---

**戻る** → [README（目次）](https://github.com/thirtypower/kgr-llm)
