---
title: "参考文献・出典（一次資料へのリンク集）"
---

> この教材の記述の根拠になった**一次資料（論文・公式ドキュメント）**を章ごとに並べたものです。
>
> **なぜこのファイルがあるのか**：LLM の解説記事には、
> 論文の主張が伝言ゲームで変形したまま広まっているものが少なくありません
> （この教材自身、後述のとおり何か所か直しています）。
> **「どこまでが論文の主張で、どこからが解説者の要約か」を追えるようにしておく**のが、
> 情報の健全性を保つ唯一の方法です。
>
> - リンクは **arXiv の abs ページ**（無料で全文が読めます）
> - arXiv 番号は **2026年8月時点で実在を確認**したものだけを載せています
> - 「読む順」の目安：★ が付いているものは、この教材を読んだ後なら**そのまま読めます**

---

## 使い方

| やりたいこと | 見る場所 |
|------------|---------|
| ある章の主張の出どころを確かめたい | 下の章別リスト |
| **この教材が一般的な説明を訂正した箇所**を知りたい | [末尾の「訂正した主張」](#この教材で明示的に訂正した主張) |
| 最新の動向を追い続けたい | [継続的に追うための情報源](#継続的に追うための情報源) |

---

## 00.5章：深層学習の基礎

| 主張・話題 | 一次資料 |
|---|---|
| Adam | [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (Kingma & Ba, 2014) |
| **AdamW**（weight decay の分離） | [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (Loshchilov & Hutter, 2017) |
| Layer Normalization | [Layer Normalization](https://arxiv.org/abs/1607.06450) (Ba et al., 2016) |

---

## 01章：NLP の基礎概念

| 主張・話題 | 一次資料 |
|---|---|
| Word2Vec（分布仮説とベクトル演算） | [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (Mikolov et al., 2013) |
| ELMo（文脈で変わる埋め込み、事前学習→微調整） | [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) (Peters et al., 2018) |
| BPE をサブワード分割に持ち込んだ論文 | [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (Sennrich et al., 2015) |

---

## 02 / 02.5章：Transformer

| 主張・話題 | 一次資料 |
|---|---|
| ★ **Transformer 本体**（Attention の式、Post-Norm、sin/cos 位置エンコーディング） | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) |
| **Pre-Norm の方が学習が安定する**理由 | [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) (Xiong et al., 2020) |
| 「知識は FFN に入っている」（FFN ＝ 連想メモリ説） | [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913) (Geva et al., 2020) |

---

## 02.6章：トークナイザ

| 主張・話題 | 一次資料 |
|---|---|
| Byte-Level BPE（256バイトから始めて未知語をなくす） | [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, Radford et al., 2019) |
| SentencePiece（**ライブラリ名**であってアルゴリズム名ではない） | [SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) (Kudo & Richardson, 2018) |
| 数字の分割方法（LLaMA-3 は最大3桁ずつ） | [Meta の tokenizer 実装](https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py)（正規表現 `\p{N}{1,3}`）／[Llama 3 論文](https://arxiv.org/abs/2407.21783) |

---

## 03章：事前学習言語モデル

| 主張・話題 | 一次資料 |
|---|---|
| ★ BERT（MLM ＋ NSP、15%マスク、80/10/10） | [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) (Devlin et al., 2018) |
| RoBERTa（NSP 削除、160GB、バッチ8K、動的マスキング） | [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) (Liu et al., 2019) |
| ALBERT（Embedding 分解・層間共有・SOP） | [ALBERT: A Lite BERT](https://arxiv.org/abs/1909.11942) (Lan et al., 2019) |
| T5（text-to-text、Span Corruption、C4 750GB） | [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683) (Raffel et al., 2019) |
| ★ GPT-3（In-Context Learning / Few-shot） | [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Brown et al., 2020) |
| LLaMA-1（RMSNorm / RoPE / SwiGLU の採用） | [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (Touvron et al., 2023) |
| LLaMA-2（GQA、34B が未公開である旨の記載） | [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) (Touvron et al., 2023) |
| LLaMA-3（15T トークン、語彙128K） | [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Grattafiori et al., 2024) |
| LLaMA-4（MoE 化、Scout/Maverick） | [Meta 公式ブログ](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)（2025）※論文なし |

---

## 04章：大規模言語モデルとは

| 主張・話題 | 一次資料 |
|---|---|
| スケーリング則（べき乗則） | [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020) |
| **Chinchilla（パラメータの約20倍のトークン）** | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022) |
| 創発能力 | [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) (Wei et al., 2022) |
| **創発への反論**（指標の不連続性のせいという主張） | [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) (Schaeffer et al., 2023) |
| Chain-of-Thought | [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903) (Wei et al., 2022) |
| RLHF（報酬モデル → PPO の3段構成） | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (InstructGPT, Ouyang et al., 2022) |
| 「SFT は引き出し方を教えるだけ」（表層的アライメント仮説） | [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) (Zhou et al., 2023) |
| **推論時スケーリング**（考える長さも性能のつまみ） | [Scaling LLM Test-Time Compute Optimally...](https://arxiv.org/abs/2408.03314) (Snell et al., 2024) |

---

## 05章：LLaMA2 の実装

| 主張・話題 | 一次資料 |
|---|---|
| RMSNorm（平均の計算を省く。7〜64%高速化） | [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (Zhang & Sennrich, 2019) |
| ★ RoPE（回転による位置表現） | [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021) |
| SwiGLU（`hidden_dim` を 2/3 にする話も原典にある） | [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (Shazeer, 2020) |
| MQA（K/V を1組に共有する極端版） | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (Shazeer, 2019) |
| GQA（MHA と MQA の中間） | [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023) |
| Flash Attention | [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022) ／ [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Dao, 2023) |

---

## 05.5章：MoE と現代アーキテクチャ

| 主張・話題 | 一次資料 |
|---|---|
| MoE の負荷分散補助損失（top-2 ルーティング） | [GShard](https://arxiv.org/abs/2006.16668) (Lepikhin et al., 2020) |
| Switch Transformer（top-1 に簡略化、容量係数） | [Switch Transformers](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021) |
| ★ Mixtral 8x7B（46.7B / 活性 12.9B、top-2 of 8） | [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (Jiang et al., 2024) |
| **共有専門家**（shared expert）と細粒度分割 | [DeepSeekMoE](https://arxiv.org/abs/2401.06066) (Dai et al., 2024) |
| ★ **MLA**（KV を潜在ベクトルに圧縮） | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (DeepSeek-AI, 2024) |
| 671B / 活性 37B、**補助損失なしの負荷分散**、FP8 学習、MTP | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (DeepSeek-AI, 2024) |
| 位置補間（PI）で文脈長を伸ばす | [Extending Context Window of LLMs via Positional Interpolation](https://arxiv.org/abs/2306.15595) (Chen et al., 2023) |
| ★ **YaRN**（周波数ごとに扱いを変える。現在の標準） | [YaRN: Efficient Context Window Extension of LLMs](https://arxiv.org/abs/2309.00071) (Peng et al., 2023) |
| 長文の「真ん中を取り落とす」現象 | [Lost in the Middle](https://arxiv.org/abs/2307.03172) (Liu et al., 2023) |
| Attention sink（文頭に注意が集まる） | [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) (Xiao et al., 2023) |
| Sliding Window Attention（全層 SWA、W=4096） | [Mistral 7B](https://arxiv.org/abs/2310.06825) (Jiang et al., 2023) |
| ★ **local : global = 5 : 1／窓 1024**、長文時の KV キャッシュ比率が約60%→15%未満 | [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) (Gemma Team, 2025) |

---

## 06章：学習フローの実践

| 主張・話題 | 一次資料 |
|---|---|
| ZeRO（Optimizer 状態 → 勾配 → 重みの順に分割） | [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (Rajbhandari et al., 2019) |
| **勾配累積の損失正規化のバグ**（micro-batch ごとに平均すると短い応答が過大評価される） | [Fixing Gradient Accumulation](https://huggingface.co/blog/gradient_accumulation)（Hugging Face, 2024.10）／[PR #34191](https://github.com/huggingface/transformers/pull/34191) |
| ★ LoRA（ΔW を低ランク近似、α/r スケーリング、B のゼロ初期化） | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021) |
| QLoRA（NF4・二重量子化・Paged Optimizer） | [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023) |
| PPO（クリッピング） | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) |
| ★ DPO（報酬モデルなしの選好学習） | [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023) |
| **DPO で chosen の尤度まで下がる現象**（likelihood displacement） | [Unintentional Unalignment: Likelihood Displacement in DPO](https://arxiv.org/abs/2410.08847) (Razin et al., 2024) |
| KTO（ペア不要・単独ラベルで学習） | [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) (Ethayarajh et al., 2024) |
| ORPO（SFT と選好学習の統合） | [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) (Hong et al., 2024) |
| **SimPO**（参照モデル不要・長さで正規化＝長さバイアス対策） | [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) (Meng et al., 2024) |

---

## 07章：LLM の応用（評価・RAG・Agent）

| 主張・話題 | 一次資料 |
|---|---|
| RAG の原型 | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020) |
| ReAct（Thought → Action → Observation） | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., 2022) |
| LLM-as-a-Judge とそのバイアス（冗長性・位置・自己好み） | [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) (Zheng et al., 2023) |
| Chatbot Arena（人間の盲検比較） | [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132) (Chiang et al., 2024) |
| MMLU | [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) (Hendrycks et al., 2020) |
| **MMLU-Pro**（MMLU の飽和への対処。10択化） | [MMLU-Pro](https://arxiv.org/abs/2406.01574) (Wang et al., 2024) |
| GSM8K | [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) (Cobbe et al., 2021) |
| HumanEval | [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (Chen et al., 2021) |
| GPQA（博士級・Google 検索でも解けない） | [GPQA: A Graduate-Level Google-Proof Q&A Benchmark](https://arxiv.org/abs/2311.12022) (Rein et al., 2023) |
| **SWE-bench**（実際の GitHub issue を直せるか） | [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) (Jimenez et al., 2023) |
| TruthfulQA | [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958) (Lin et al., 2021) |
| **長文評価**（宣伝された文脈長と実効長は違う） | [RULER](https://arxiv.org/abs/2404.06654) (Hsieh et al., 2024) ／ [LongBench](https://arxiv.org/abs/2308.14508) (Bai et al., 2023) |

---

## 07.5章：推論を速くする

| 主張・話題 | 一次資料 |
|---|---|
| ★ **PagedAttention / vLLM**（KV キャッシュの断片化解消、continuous batching） | [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (Kwon et al., SOSP 2023) |
| **投機的デコード**（出力分布が元のモデルと一致する証明つき） | [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2022) |
| Medusa（下書きモデルを使わない変種） | [Medusa: Simple LLM Inference Acceleration Framework](https://arxiv.org/abs/2401.10774) (Cai et al., 2024) |
| GPTQ（誤差を補正しながら int4 量子化） | [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323) (Frantar et al., 2022) |
| AWQ（活性値から重要な重みを見つけて守る） | [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) (Lin et al., 2023) |
| 実装・運用の詳細 | [vLLM 公式ドキュメント](https://docs.vllm.ai/) ／ [llama.cpp](https://github.com/ggml-org/llama.cpp) |

---

## 08章：強化学習

| 主張・話題 | 一次資料 |
|---|---|
| ★ **GRPO**（価値モデル不要、グループ内の相対評価） | [DeepSeekMath](https://arxiv.org/abs/2402.03300) (Shao et al., 2024) |
| ★ **RLVR で推論能力が創発する** | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948) (DeepSeek-AI, 2025) |
| **GRPO の長さバイアス／std 正規化の偏り**（Dr. GRPO） | [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) (Liu et al., 2025) |
| **トークン単位の損失集計・clip-higher・dynamic sampling** | [DAPO: An Open-Source LLM RL System at Scale](https://arxiv.org/abs/2503.14476) (Yu et al., 2025) |
| Search-R1（検索するかどうかを RL で学ぶ、観測マスク） | [Search-R1](https://arxiv.org/abs/2503.09516) (Jin et al., 2025) |
| ReTool（コード実行を「いつ使うか」から学ぶ） | [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536) (Feng et al., 2025) |

---

## この教材で明示的に訂正した主張

「よく言われているが、一次資料に照らすと不正確」なものを、この教材では明示的に直しています。
**他の解説記事を読むときの注意点としても使えます。**

| よく見る説明 | 実際 | この教材の該当箇所 |
|---|---|---|
| 「RoPE は長文にそのまま外挿できる」 | **素の RoPE は学習長を超えると崩れる。** PI / NTK-aware / YaRN のような細工が必須 | [05章 5.1.3](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/05-build-your-own-llm) の ⚠️、[05.5章 5.5.3](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/055-moe-modern-arch) |
| 「最近のモデルは数字を1桁ずつに分割する」 | **分かれている。** LLaMA-1/2・Gemma は1桁ずつ、GPT-4・LLaMA-3 は**最大3桁ずつ** | [02.6章 3.4](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/026-tokenizer) |
| 「`8x7B` は 7B × 8 = 56B」 | **46.7B。** MoE 化されるのは FFN だけで、Attention と Embedding は共有される | [05.5章 5.5.1](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/055-moe-modern-arch) |
| 「MoE は軽い」 | **計算量は軽いが、メモリは重い**（全専門家を常駐させる必要がある） | [05.5章 5.5.1](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/055-moe-modern-arch) |
| 「GQA / MLA は計算量を削減する」 | **削減するのは保存量（KV キャッシュ）。計算量は変わらない**（使う直前に復元・複製するため） | [05章 5.1.4](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/05-build-your-own-llm)、[05.5章 5.5.2](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/055-moe-modern-arch) |
| 「KV キャッシュを入れれば1トークンの値段は一定」 | **一定にならない。** Attention 側（過去の K/V の読み出し）は原理的に消せない | [07.5章 7.5.2](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/075-fast-inference)（実測つき） |
| 「MMLU が 88点なら高性能」 | **MMLU は飽和済み**（上位モデルが軒並み90%超）。2026年は GPQA-Diamond / SWE-bench Verified / LiveCodeBench 等で測る | [07章 7.1.1](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/07-applications) |
| 「DPO は RLHF の完全な代替」 | **長さバイアス・オフラインの限界がある。** 実務は「DPO で土台 → on-policy RL で仕上げ」に回帰しつつある | [06章 6.4.2](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/06-training-pipeline) |
| 「GRPO の式はそのまま使えばよい」 | **元の式には2つの偏り**（長さ正規化・std 正規化）があり、Dr. GRPO / DAPO で修正された | [08章 8.1.4](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/08-reinforcement-learning) |
| 「Chinchilla の20倍が最適」 | **学習コストだけを最小にする点。** 推論コストを含めると、はるかに多く学習した方が得 | [04章 4.2.1](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/04-what-is-llm) |
| 「Flash Attention は計算量を減らす」 | **減らすのはメモリ移動（HBM ↔ SRAM の往復）。** 計算量は T² のままで、出力は素朴な実装と一致する（近似ではない） | [05章 5.1.4](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/05-build-your-own-llm) |
| 「QLoRA は 4bit だから速い」 | **速くならない。** メモリを削るための技術で、bf16 に戻す処理が挟まるぶん **bf16 の LoRA より遅い** | [06章 6.3.4](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/06-training-pipeline) |
| 「創発は規模がある閾値を超えると必ず起きる」 | **測り方に依存する。** 完全一致のような不連続な指標は、なだらかな改善を崖に見せる（Schaeffer et al.） | [04章 4.1.2](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/04-what-is-llm)（2指標の比較図） |
| 「マスク付き Attention はカンニング防止のため」 | 正しいが**それだけではない**。本質は **1回の forward で全位置ぶんの学習をする**ためで、これが無いと事前学習の計算量が T 倍になる | [02章 2.1.7](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/02-transformer-attention) |
| 「勾配累積は大きいバッチと完全に等価」 | **応答の長さがバラバラだと等価にならない**（micro-batch ごとに平均すると短い応答が過大評価される）。2024年に Hugging Face 側で修正された | [05章（後編）5.3.3](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/05-2-pretrain-your-llm) |

> 🔑 **共通する教訓**：この表の項目はほとんどが
> **「何を最小化した結論なのか」「何が減って何が減らないのか」を落とした要約**から生まれています。
> 論文の主張を読むときは、**前提条件と、測っている量**を必ず確認してください。

---

## 継続的に追うための情報源

LLM は動きが速いので、**教材は必ず古くなります**。追い続けるための定番を挙げます。

| 種類 | 情報源 |
|------|-------|
| アーキテクチャの比較解説 | [Sebastian Raschka のブログ / LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)（新モデルの構成を図で比較してくれる） |
| 論文の動向まとめ | [Ahead of AI](https://magazine.sebastianraschka.com/)、[Hugging Face Daily Papers](https://huggingface.co/papers) |
| 人間評価のランキング | [LMArena（旧 LMSYS Chatbot Arena）](https://lmarena.ai/) |
| オープンモデルの自動評価 | [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) |
| **日本語 LLM** | [awesome-japanese-llm（llm-jp）](https://github.com/llm-jp/awesome-japanese-llm)、[Nejumi LLM リーダーボード](https://wandb.ai/wandb-japan/llm-leaderboard) |
| 推論の実装 | [vLLM のブログ](https://blog.vllm.ai/)、[llama.cpp の議論](https://github.com/ggml-org/llama.cpp/discussions) |

> 💡 **追い方のコツ**：新しい手法が出たら、まず
> **「何を節約しようとしているのか（計算量／メモリ／データ／人手）」** を1文で書いてみてください。
> この教材の 05.5章・07.5章がその形で整理されているのは、
> **その1文が言えれば、細かい実装を知らなくても位置づけが分かる**からです。

---

**戻る** → [README（目次）](https://github.com/thirtypower/kgr-llm) ／ [用語集](https://zenn.dev/thirtypower/books/kgr-llm-guide/viewer/appendix-glossary)
