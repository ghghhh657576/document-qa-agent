# -*- coding: utf-8 -*-
"""切块质量离线评估（不需要 API Key）。

回答两个问题：
1. 朴素硬切（每 n 字切一刀）和标点感知切分（split_every）相比，块边界切断 jieba 词的比例是多少？
2. 默认 n=500 时切出多少块、长度分布如何、有没有重复块？

指标定义（与 README「切分感知标点」一节的表述对应）：
- 边界切断率：块边界落在某个多字词内部，这类边界数 / 全部边界数
- 块切中率：块至少有一端切断词，这类块数 / 全部块数

用法：python eval/chunk_eval.py
"""
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jieba  # noqa: E402
import rag  # noqa: E402

DATA = ROOT / "data"


def word_spans(text):
    spans = []
    pos = 0
    for w in jieba.cut(text):
        spans.append((pos, pos + len(w), w))
        pos += len(w)
    return spans


def naive_split(s, n):
    return [s[i:i + n] for i in range(0, len(s), n)]


def boundary_stats(chunks, text):
    """返回 (切断的边界数, 边界总数, 切中词的块数, 块数)。"""
    spans = [s for s in word_spans(text) if len(s[2]) > 1 and s[2].strip()]
    acc = 0
    cut_bounds = 0
    cut_chunks = 0
    bounds_total = 0
    for i, c in enumerate(chunks):
        start, end = acc, acc + len(c)
        acc = end
        if i == len(chunks) - 1:
            break
        bounds_total += 1
        if _inside_any(end, spans):
            cut_bounds += 1
            cut_chunks += 1
        elif _inside_any(start, spans):
            cut_chunks += 1
    return cut_bounds, bounds_total, cut_chunks, len(chunks)


def _inside_any(pos, spans):
    for start, end, _w in spans:
        if start < pos < end:
            return True
    return False


def main():
    files = sorted(DATA.glob("*.txt")) + sorted(DATA.glob("*.md"))
    if not files:
        raise SystemExit(f"没有在 {DATA} 找到知识库文件")

    print("=== 1. 切块边界质量：朴素硬切 vs 标点感知切分 ===")
    for n in (500, 300):
        tb = tn = tc = tc_total = sb = sn = sc = sc_total = 0
        naive_chunks = smart_chunks = 0
        for f in files:
            text = f.read_text(encoding="utf-8")
            naive = naive_split(text, n)
            smart = rag.split_every(text, n)
            naive_chunks += len(naive)
            smart_chunks += len(smart)
            b1, n1, c1, m1 = boundary_stats(naive, text)
            b2, n2, c2, m2 = boundary_stats(smart, text)
            tb += b1; tn += n1; tc += c1; tc_total += m1
            sb += b2; sn += n2; sc += c2; sc_total += m2
        print(f"n={n}  块数 naive={naive_chunks}  smart={smart_chunks}")
        print(f"      切断边界 naive {tb}/{tn} = {tb / tn:.1%}   "
              f"smart {sb}/{sn} = {sb / sn:.1%}")
        print(f"      切中词块 naive {tc}/{tc_total} = {tc / tc_total:.1%}   "
              f"smart {sc}/{sc_total} = {sc / sc_total:.1%}")

    print()
    print("=== 2. n=500 时的切块统计 ===")
    chunks = rag.load_all(str(DATA), 500)
    lens = [len(c["content"]) for c in chunks]
    dup = len(chunks) - len({c["content"] for c in chunks})
    print(f"块数 = {len(chunks)}")
    print(f"长度 min / 中位数 / max = {min(lens)} / {statistics.median(lens):.0f} / {max(lens)}")
    print(f"短于 100 字的块 = {sum(1 for x in lens if x < 100)}，超长（>600）块 = {sum(1 for x in lens if x > 600)}")
    print(f"重复内容块 = {dup}")
    print()
    print("说明：这里的数据是仓库自带的 3 份资料，换语料后数字会变；指标定义见本文件开头注释。")


if __name__ == "__main__":
    main()
