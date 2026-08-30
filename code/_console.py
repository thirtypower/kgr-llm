"""Windows のコンソール（cp932）でも日本語・記号が化けないようにするための小道具。

各スクリプトの先頭で `import _console` するだけで、標準出力が UTF-8 になります。
それでも文字化けする場合は、実行前に `chcp 65001` を実行するか、
`python -X utf8 xxx.py` として起動してください。
"""

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:      # 古い Python / 特殊な環境では何もしない
        pass


def title(s: str) -> None:
    """章タイトルを見やすく出す。"""
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


def sub(s: str) -> None:
    """小見出し。"""
    print("\n--- " + s + " " + "-" * max(0, 66 - len(s)))
