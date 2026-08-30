"""
07. LoRA 微調整を1回通す（06章 6.3 節に対応）
======================================================================
    python 07_lora_sft.py                    # dolly-15k-ja から 500 件で LoRA SFT
    python 07_lora_sft.py --samples 100      # データをさらに絞った高速版
    python 07_lora_sft.py --samples 20 --max-steps 5   # 動作確認だけ（数分）

Qwen2.5-0.5B-Instruct に、日本語の指示応答データ（databricks-dolly-15k-ja）を
LoRA で微調整し、**学習の前後で同じ質問への応答がどう変わるか**を見比べます。

このスクリプトが 06章の本文と対応する箇所
----------------------------------------------------------------------
    6.2.3  損失マスク（プロンプト部分を -100 にする）  -> build_example()
    6.2.3  パディング（batch をまとめるための詰め物）   -> collate()
    6.3.3  LoraConfig / get_peft_model                -> main() の②
    6.3.3  Trainer で学習 → save_pretrained           -> main() の④

必要なもの
----------------------------------------------------------------------
    pip install transformers datasets peft accelerate

    GPU:   VRAM 6GB 程度から動きます（0.5B + LoRA。bf16/fp16 使用時）。
           Colab 無料枠の T4 (16GB) なら余裕です。
    CPU:   一応動きますが非常に遅いので --samples 20 --max-steps 5 推奨。

Google Colab での実行手順（3行）
----------------------------------------------------------------------
    !pip install -q transformers datasets peft accelerate
    !git clone https://github.com/thirtypower/KGR-LLM.git
    !cd KGR-LLM/kgr-llm/code && python 07_lora_sft.py --samples 500

対応ドキュメント: ../06-学習フローの実践.md（6.3 節）
"""

import argparse
import sys

from _console import title, sub

# ----------------------------------------------------------------------
# 依存ライブラリの確認（無い場合は分かりやすく案内して終了する）
# ----------------------------------------------------------------------
try:
    import torch
except ImportError:
    sys.exit("\n  PyTorch が必要です:  pip install torch\n")

_missing = []
for _mod in ("transformers", "datasets", "peft"):
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_mod)
if _missing:
    sys.exit(
        f"\n  このスクリプトには {' / '.join(_missing)} が必要です。\n"
        f"  次のコマンドでインストールしてください:\n\n"
        f"      pip install transformers datasets peft accelerate\n\n"
        f"  （GPU が無い場合は Google Colab 推奨。手順はこのファイル冒頭の docstring 参照）\n")

from datasets import load_dataset                                    # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model                # noqa: E402
from transformers import (AutoModelForCausalLM, AutoTokenizer,       # noqa: E402
                          Trainer, TrainingArguments)

torch.manual_seed(0)

# 学習前後の比較に使う質問（データセットに直接は含まれない、素朴な日本語の指示）
EVAL_PROMPTS = [
    "日本で一番高い山は何ですか？",
    "電子レンジでゆで卵を作ってはいけない理由を教えてください。",
    "「猫の手も借りたい」とはどういう意味ですか？",
]


# ======================================================================
# データの準備（06章 6.2.3 の「損失マスク」をそのまま実装）
# ======================================================================
def build_example(item, tokenizer, max_len):
    """dolly-15k-ja の1件を (input_ids, labels) にする。

    プロンプト部分（system/user + assistant ヘッダ）のラベルは -100 にして
    損失計算から除外し、**応答部分だけを学習対象にする**（6.2.3 の損失マスク）。
    """
    question = item["instruction"]
    if item.get("input"):                     # 参考文脈が付いている問題もある
        question = question + "\n\n" + item["input"]

    messages = [{"role": "user", "content": question}]
    # add_generation_prompt=True で「<|im_start|>assistant\n」までを含める
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True)
    answer_ids = tokenizer(item["output"] + tokenizer.eos_token,
                           add_special_tokens=False).input_ids

    input_ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    return {"input_ids": input_ids, "labels": labels}


def make_collator(pad_token_id):
    """バッチ内の最長サンプルに合わせてパディングする data collator。

    6.2.3 の SupervisedDataset に足りなかった「詰め物」がこれです。
    input_ids は pad_token_id、labels は -100（損失計算外）、
    attention_mask は 0（Attention が見ない）で埋めます。
    """
    def collate(batch):
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids, labels, attn = [], [], []
        for ex in batch:
            pad = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_token_id] * pad)
            labels.append(ex["labels"] + [-100] * pad)
            attn.append([1] * len(ex["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }
    return collate


# ======================================================================
# 生成（学習前後の比較用）
# ======================================================================
@torch.no_grad()
def ask(model, tokenizer, question, max_new_tokens=120):
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(
        inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def show_responses(model, tokenizer, label):
    sub(label)
    model.eval()
    answers = []
    for q in EVAL_PROMPTS:
        a = ask(model, tokenizer, q)
        answers.append(a)
        print(f"  Q: {q}")
        print(f"  A: {a[:200]}{'...' if len(a) > 200 else ''}\n")
    return answers


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="0.5B モデルの LoRA SFT を1回通す")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dataset", default="kunishou/databricks-dolly-15k-ja")
    ap.add_argument("--samples", type=int, default=500,
                    help="使う学習データ件数（既定 500。全 15k 件使うなら -1）")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="指定するとこのステップ数で打ち切る（動作確認用）")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="LoRA はフル SFT より高めが定番（6.3.4 参照）")
    ap.add_argument("--output", default="./lora_adapter")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  ⚠ GPU が見つかりません。CPU では非常に遅いので"
              " --samples 20 --max-steps 5 程度を推奨します")

    # ------------------------------------------------------------------
    title("① ベースモデルとトークナイザの読み込み")
    # ------------------------------------------------------------------
    print(f"  モデル: {args.model}（初回はダウンロードに数分かかります）")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = (torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float16 if device == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype).to(device)
    print(f"  デバイス: {device} / dtype: {dtype}")

    # 学習「前」の応答を記録しておく（あとで見比べる）
    before = show_responses(model, tokenizer, "学習前の応答（ベースモデルそのまま）")

    # ------------------------------------------------------------------
    title("② LoRA アダプタの取り付け（6.3.3 の本文コード）")
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                       # ランク
        lora_alpha=16,             # スケーリング（通常 r の2倍）
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],   # Attention のみ
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("  ★ 元の重みは全部凍結され、上の trainable% だけが学習対象です")

    # ------------------------------------------------------------------
    title("③ データセットの準備（6.2.3 の損失マスク）")
    # ------------------------------------------------------------------
    print(f"  データ: {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    if args.samples > 0:
        ds = ds.shuffle(seed=0).select(range(min(args.samples, len(ds))))
    ds = ds.map(lambda ex: build_example(ex, tokenizer, args.max_len),
                remove_columns=ds.column_names)
    print(f"  学習データ: {len(ds)} 件（プロンプト部分は -100 でマスク済み）")

    # ------------------------------------------------------------------
    title("④ 学習（Trainer に任せる）")
    # ------------------------------------------------------------------
    train_args = TrainingArguments(
        output_dir="./sft_out",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",            # チェックポイントは最後に自分で保存する
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=make_collator(tokenizer.pad_token_id or tokenizer.eos_token_id),
    )
    trainer.train()

    model.save_pretrained(args.output)     # ★ 保存されるのはアダプタ数MBだけ
    print(f"\n  LoRA アダプタを保存: {args.output}（ベースモデル本体は含まれない）")

    # ------------------------------------------------------------------
    title("⑤ 学習後の応答と見比べる")
    # ------------------------------------------------------------------
    after = show_responses(model, tokenizer, "学習後の応答（LoRA 適用済み）")

    sub("まとめ")
    changed = sum(1 for b, a in zip(before, after) if b != a)
    print(f"""  {len(EVAL_PROMPTS)} 問中 {changed} 問で応答が変化しました。
  0.5B + 数百件の SFT なので劇的には変わりませんが、
  「口調・答え方の形式」が dolly-ja のスタイルに寄っていれば成功です。
  ここを数万件・数エポックにすると変化がはっきりします。

  推論だけやり直すときは（6.3.3 の本文コード）:
      from peft import PeftModel
      base  = AutoModelForCausalLM.from_pretrained("{args.model}")
      model = PeftModel.from_pretrained(base, "{args.output}")""")


if __name__ == "__main__":
    main()
