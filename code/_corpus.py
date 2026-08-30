"""学習用の日本語ミニコーパス（外部ダウンロード不要）。

なぜ自前のコーパスを使うのか
--------------------------------------------------------------------
本物の Wikipedia などで学習すると、「モデルがちゃんと学習できたのか」を
目で見て判断するしかありません。判断できないと、実装のバグなのか
データ量不足なのか永久に切り分けられません。

そこでこの教材では、まず **文法規則が完全に分かっているコーパス** を使います。
規則が既知なので `is_valid()` で生成文の正しさを機械的に採点でき、
「自作モデルは本当に学習したのか」を数値で確認できます。

    正答率 90% 以上  → 実装は正しい。次は本物のデータへ
    正答率 30% 前後  → どこかにバグがある（学習不足ではない）

規則
--------------------------------------------------------------------
    1. {動物}が{食べ物}を食べる。
    2. {人}は{場所}へ行く。
    3. {人}が{動物}を見る。
    4. {場所}に{動物}がいる。
    5. {人}は{食べ物}を買う。

「猫が学校を食べる」のような文は絶対に出てきません。
つまりモデルは「が の後ろに来られる語」「を の後ろに来られる語」を
文脈から学ばなければ正解できません。これは立派な文法学習です。
"""

import random
import re
from typing import List

ANIMALS = ["猫", "犬", "鳥", "馬", "兎", "狐"]
FOODS = ["魚", "肉", "米", "草", "豆", "菓子"]
PEOPLE = ["太郎", "花子", "先生", "学生", "母", "父"]
PLACES = ["学校", "公園", "駅", "海", "山", "店"]

TEMPLATES = [
    ("{a}が{f}を食べる。", ("a", "f")),
    ("{p}は{l}へ行く。", ("p", "l")),
    ("{p}が{a}を見る。", ("p", "a")),
    ("{l}に{a}がいる。", ("l", "a")),
    ("{p}は{f}を買う。", ("p", "f")),
]

_SLOTS = {"a": ANIMALS, "f": FOODS, "p": PEOPLE, "l": PLACES}

# 採点用の正規表現（テンプレートと 1 対 1 で対応）
_PATTERNS = [
    re.compile(rf"^(?:{'|'.join(ANIMALS)})が(?:{'|'.join(FOODS)})を食べる。$"),
    re.compile(rf"^(?:{'|'.join(PEOPLE)})は(?:{'|'.join(PLACES)})へ行く。$"),
    re.compile(rf"^(?:{'|'.join(PEOPLE)})が(?:{'|'.join(ANIMALS)})を見る。$"),
    re.compile(rf"^(?:{'|'.join(PLACES)})に(?:{'|'.join(ANIMALS)})がいる。$"),
    re.compile(rf"^(?:{'|'.join(PEOPLE)})は(?:{'|'.join(FOODS)})を買う。$"),
]


def make_sentences(n: int = 20000, seed: int = 0) -> List[str]:
    """規則に従った文を n 個作る。"""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        tmpl, slots = TEMPLATES[rng.randrange(len(TEMPLATES))]
        values = {s: rng.choice(_SLOTS[s]) for s in slots}
        out.append(tmpl.format(**values))
    return out


def make_text(n: int = 20000, seed: int = 0) -> str:
    """文を改行でつないだ 1 本のテキストにする（事前学習用）。"""
    return "\n".join(make_sentences(n, seed)) + "\n"


def is_valid(sentence: str) -> bool:
    """生成された 1 文が文法規則に合っているかを判定する。"""
    s = sentence.strip()
    return any(p.match(s) for p in _PATTERNS)


def score(text: str):
    """生成テキスト全体を採点する。 -> (正しい文の数, 判定した文の総数, 正答率)"""
    lines = [ln for ln in text.replace("。", "。\n").split("\n") if ln.strip()]
    if not lines:
        return 0, 0, 0.0
    ok = sum(1 for ln in lines if is_valid(ln))
    return ok, len(lines), ok / len(lines)


# 語彙の全体像（ドキュメント用）
def describe() -> str:
    return (
        f"動物 {len(ANIMALS)} 語 / 食べ物 {len(FOODS)} 語 / "
        f"人 {len(PEOPLE)} 語 / 場所 {len(PLACES)} 語、"
        f"文型 {len(TEMPLATES)} 種類 → 作れる文は "
        f"{sum(len(_SLOTS[s[0]]) * len(_SLOTS[s[1]]) for _, s in TEMPLATES)} 通り"
    )


if __name__ == "__main__":
    import _console  # noqa: F401  （UTF-8 出力にする）

    print(describe())
    print("\nサンプル:")
    for s in make_sentences(8, seed=1):
        print("  ", s)
    print("\n採点のテスト:")
    for s in ["猫が魚を食べる。", "猫が学校を食べる。", "太郎は公園へ行く。", "あああ"]:
        print(f"   {s:20s} -> {'正しい' if is_valid(s) else '間違い'}")
