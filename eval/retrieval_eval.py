# -*- coding: utf-8 -*-
"""BM25 检索离线评估（不需要 API Key）。

用 12 个有标准答案的问题，检查「期望关键词是否出现在检索结果里」，输出 top-1 / top-3 / MRR@3。
另外检查三个边界行为：语料外问题是否 0 命中、英文大小写是否一致、纯停用词查询是否 0 命中。

注意：用例的标准答案是「关键词出现在某个块里」，只衡量检索能否把正确块召回，
不代表生成答案一定正确（那是 agent 层的事）。语料里没有的词（如“引用校验”“缓存命中率”）
不能做题干用例，故未列入。

用法：python eval/retrieval_eval.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag  # noqa: E402

CASES = [
    ("装饰器怎么用", "装饰器"),
    ("双指针", "双指针"),
    ("滑动窗口", "滑动窗口"),
    ("二分查找", "二分查找"),
    ("哈希表", "哈希表"),
    ("Function Calling 是什么", "Function Calling"),
    ("RAG 和 Function Calling 的区别", "Function Calling"),
    ("简历上的项目怎么写", "简历"),
    ("什么时候投实习", "投递"),
    ("BM25 和向量检索的区别", "BM25"),
    ("多轮对话怎么实现的", "多轮对话"),
    ("怎么复用代码", "装饰器"),
]


def main():
    chunks = rag.load_all(str(ROOT / "data"), 500)
    bm25 = rag.build_index(chunks)
    print(f"知识库：{len(chunks)} 块（n=500）")
    print()
    print(f"{'top-1':<6}{'top-3':<6}{'名次':<6}{'问题':<34}期望关键词")
    top1 = top3 = mrr = 0.0
    for q, kw in CASES:
        _text, hits = rag.search(q, chunks, bm25)
        idxs = [i for i, _s in hits]
        rank = None
        for r, i in enumerate(idxs, start=1):
            if kw in chunks[i]["content"]:
                rank = r
                break
        ok1 = rank == 1
        ok3 = rank is not None
        top1 += ok1
        top3 += ok3
        mrr += (1.0 / rank) if rank else 0.0
        print(f"{'OK' if ok1 else '-':<6}{'OK' if ok3 else '-':<6}{str(rank):<6}{q:<34}{kw}")
        print("       命中块：" + ", ".join(
            f"#{i}（{chunks[i]['source']}，{s:.2f}）" for i, s in hits))

    n = len(CASES)
    print()
    print(f"top-1 命中率 = {top1:.0f}/{n} = {top1 / n:.0%}")
    print(f"top-3 命中率 = {top3:.0f}/{n} = {top3 / n:.0%}")
    print(f"MRR@3 = {mrr / n:.2f}")

    print()
    print("=== 边界行为 ===")
    for q in ("如何做红烧肉", "量子计算", "今天天气怎么样"):
        text, hits = rag.search(q, chunks, bm25)
        print(f"语料外问题 {q!r} -> {len(hits)} 命中，回答={text!r}")
    up = [i for i, _ in rag.search("BM25", chunks, bm25)[1]]
    lo = [i for i, _ in rag.search("bm25", chunks, bm25)[1]]
    print(f"大小写：BM25 -> {up}，bm25 -> {lo}，{'一致' if up == lo else '不一致'}")
    for q in ("的", "怎么"):
        print(f"纯停用词 {q!r} -> {len(rag.search(q, chunks, bm25)[1])} 命中")


if __name__ == "__main__":
    main()
