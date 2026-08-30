"""Byte-Level BPE トークナイザの実装本体。

解説と実行例は 03_bpe_tokenizer.py にあります（このモジュールを import して使います）。
ドキュメント: ../02.6-トークナイザを理解する.md
"""

import json
import re
from collections import Counter
from typing import Dict, List, Tuple

# ======================================================================
# 事前分割（pre-tokenization）
# ----------------------------------------------------------------------
# BPE をいきなり文章全体に掛けると、「。猫」のように意味的に無関係な組が
# 結合されてしまいます。そこで先にざっくり塊に切っておきます。
# 英語では空白で切るのが定番ですが、日本語は空白が無いので
# 文字種（ひらがな/カタカナ/漢字/英数）の切れ目で分けます。
# ======================================================================
_PRETOKEN_RE = re.compile(
    r"[ぁ-ゟ]+"                 # ひらがな
    r"|[ァ-ヿー]+"              # カタカナ
    r"|[一-鿿々〆ヶ]+"           # 漢字
    r"|[A-Za-z]+"               # 英字
    r"|[0-9]+"                  # 数字
    r"|\s+"                     # 空白
    r"|."                       # それ以外は 1 文字ずつ
)


def pretokenize(text: str) -> List[str]:
    return _PRETOKEN_RE.findall(text)


def render(data: bytes) -> str:
    """トークンの中身を人間が読める形にする。

    BPE の途中経過には「UTF-8 として不完全なバイト列」が普通に現れます
    （例: 「猫」= E7 8C AB のうち E7 8C だけ）。それは文字として表示できないので
    <0xE7 0x8C> のように 16 進で見せます。ここを誤魔化さないことが理解の近道です。
    """
    try:
        return "'" + data.decode("utf-8") + "'"
    except UnicodeDecodeError:
        return "<" + " ".join(f"0x{b:02X}" for b in data) + ">"


class ByteLevelBPE:
    """Byte-Level BPE トークナイザ（教材用の最小実装）。

    語彙 = [特殊トークン] + [256 個のバイト] + [マージで作った新トークン]
    """

    def __init__(self, special_tokens: List[str] = None):
        self.special_tokens = special_tokens or ["<pad>", "<s>", "</s>",
                                                 "<|im_start|>", "<|im_end|>"]
        # id -> bytes
        self.vocab: Dict[int, bytes] = {}
        # (id_a, id_b) -> 新しい id   ※学習で得られるマージ規則
        self.merges: Dict[Tuple[int, int], int] = {}
        self.special_to_id: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # 訓練
    # ------------------------------------------------------------------
    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert vocab_size > 256 + len(self.special_tokens)

        # --- 語彙の土台を作る -----------------------------------------
        # ① 特殊トークン
        self.special_to_id = {t: i for i, t in enumerate(self.special_tokens)}
        n_special = len(self.special_tokens)
        for t, i in self.special_to_id.items():
            self.vocab[i] = t.encode("utf-8")
        # ② 0〜255 の全バイト（これで「未知語」が原理的に存在しなくなる）
        for b in range(256):
            self.vocab[n_special + b] = bytes([b])
        next_id = n_special + 256
        self._byte_base = n_special      # バイト b の id は _byte_base + b

        # --- 訓練データを「塊 -> 出現回数」の形にする -----------------
        # 同じ塊を何度も処理しないための高速化。BPE の結果は変わりません。
        chunk_freq = Counter(pretokenize(text))
        # 各塊を「バイト id の並び」に変換
        corpus: List[Tuple[List[int], int]] = [
            ([self._byte_base + b for b in chunk.encode("utf-8")], freq)
            for chunk, freq in chunk_freq.items()
        ]

        n_merges = vocab_size - next_id
        for step in range(n_merges):
            # --- 隣り合うペアの出現回数を数える ------------------------
            pair_freq = Counter()
            for ids, freq in corpus:
                for a, b in zip(ids, ids[1:]):
                    pair_freq[(a, b)] += freq
            if not pair_freq:
                break

            # --- 最頻ペアを 1 つの新トークンにする --------------------
            best_pair, best_count = pair_freq.most_common(1)[0]
            if best_count < 2:           # もう結合する価値がない
                break
            new_id = next_id
            next_id += 1
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose and step < 12:
                print(f"  マージ {step + 1:3d}: "
                      f"{render(self.vocab[best_pair[0]]):>16s}"
                      f" + {render(self.vocab[best_pair[1]]):<16s}"
                      f" -> {render(self.vocab[new_id]):<18s}"
                      f" (出現 {best_count} 回)")

            # --- コーパス中の該当ペアを全部置き換える ------------------
            corpus = [(self._merge_ids(ids, best_pair, new_id), freq)
                      for ids, freq in corpus]

        return self

    @staticmethod
    def _merge_ids(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    # 符号化・復号
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for chunk in pretokenize(text):
            piece = [self._byte_base + b for b in chunk.encode("utf-8")]
            # 学習したマージ規則を「学習した順（＝id が小さい順）」に適用する。
            # 順番を守らないと訓練時と違うトークン列になるので注意。
            while len(piece) >= 2:
                candidates = {}
                for a, b in zip(piece, piece[1:]):
                    if (a, b) in self.merges:
                        candidates[(a, b)] = self.merges[(a, b)]
                if not candidates:
                    break
                pair = min(candidates, key=candidates.get)   # 一番早く学習したマージ
                piece = self._merge_ids(piece, pair, self.merges[pair])
            ids.extend(piece)
        return ids

    def decode(self, ids: List[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.vocab)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "special_tokens": self.special_tokens,
                "vocab": {str(k): list(v) for k, v in self.vocab.items()},
                "merges": [[a, b, c] for (a, b), c in self.merges.items()],
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ByteLevelBPE":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        t = cls(d["special_tokens"])
        t.vocab = {int(k): bytes(v) for k, v in d["vocab"].items()}
        t.merges = {(a, b): c for a, b, c in d["merges"]}
        t.special_to_id = {s: i for i, s in enumerate(d["special_tokens"])}
        t._byte_base = len(d["special_tokens"])
        return t


