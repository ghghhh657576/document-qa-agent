# -*- coding: utf-8 -*-
"""引用校验兜底的行为验证（不需要 API Key，用假的模型客户端）。

验证 README 里承诺的四件事：
1. 本轮没检索资料、回答里也没有块引用 → 拦截，返回兜底话术
2. 引用不存在的块号 → 回灌真实块号重试一次；重试后仍编造 → 拦截，不把不可核对的内容返回
3. 正常检索 + 引用真实块号 → 正常放行
4. 纯出题请求（调用 generate_quiz）的回答没有块引用 → 放行，不误拦

用法：python eval/citation_gate.py   （退出码 0 表示全部通过）
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DEEPSEEK_API_KEY", "sk-offline-test")
import rag  # noqa: E402

OUT = []


def log(line=""):
    print(line)
    OUT.append(line)


def check(name, ok, detail=""):
    log(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    return ok


class FakeUsage:
    prompt_tokens = 100
    completion_tokens = 20
    prompt_cache_hit_tokens = 0


def _response(message):
    return type("Resp", (), {"usage": FakeUsage(), "choices": [type("C", (), {"message": message})()]})()


def _msg(content=None, tool_calls=None):
    return type("Msg", (), {"content": content, "tool_calls": tool_calls,
                            "model_dump": lambda self: {"role": "assistant", "content": content}})()


def _tool_call(name, arguments):
    fn = type("Fn", (), {"name": name, "arguments": arguments})()
    return type("TC", (), {"id": "call_1", "function": fn})()


class FakeClient:
    def __init__(self, builders):
        self.builders = list(builders)

        class Completions:
            def __init__(self, outer):
                self.outer = outer

            def create(self, **kwargs):
                if not self.outer.builders:
                    raise AssertionError("模型被调用了多余的次数")
                return _response(self.outer.builders.pop(0)(kwargs.get("messages")))

        self.chat = type("Chat", (), {"completions": Completions(self)})()


def run_with(builders):
    client = FakeClient(builders)
    rag.OpenAI = lambda api_key=None, base_url=None: client
    return client


def main():
    chunks = rag.load_all(str(ROOT / "data"), 500)
    bm25 = rag.build_index(chunks)
    ok = True

    log("=== 1. 没检索、没引用 ===")
    run_with([lambda _m: _msg(content="巴黎是法国的首都。")])
    answer, log_lines = rag.agent("法国的首都是哪里？", chunks, bm25, [])
    ok &= check("回答被拦截为兜底话术", answer == rag.FALLBACK_NO_SOURCE, answer[:24])
    ok &= check("日志记录了拦截原因", any("已拦截" in x for x in log_lines))

    log()
    log("=== 2. 编造块号，重试后仍然编造 ===")
    run_with([
        lambda _m: _msg(content="答案……（块 #888，x.txt）"),
        lambda _m: _msg(content="答案……（块 #999，y.txt）"),
    ])
    answer, log_lines = rag.agent("随便问问", chunks, bm25, [])
    ok &= check("回答被拦截为兜底话术", answer == rag.FALLBACK_CITATION, answer[:24])
    ok &= check("日志记录了重试后仍失败", any("已重试过一次" in x for x in log_lines))

    log()
    log("=== 3. 正常检索 + 引用真实块号 ===")
    _text, hits = rag.search("装饰器", chunks, bm25)
    idx = hits[0][0]
    good = f"装饰器可以复用一个函数的逻辑（块 #{idx}，{chunks[idx]['source']}）"
    run_with([
        lambda _m: _msg(tool_calls=[_tool_call("search_material", '{"query": "装饰器"}')]),
        lambda _m: _msg(content=good),
    ])
    answer, _logs = rag.agent("装饰器是什么？", chunks, bm25, [])
    ok &= check("正常回答不被误拦", answer == good, answer[:24])

    log()
    log("=== 4. 纯出题请求（出题结果没有块引用）===")
    quiz_text = ("1. 装饰器的作用是什么？\n答案：在不改原函数的前提下增加功能。\n"
                 "解析：装饰器是接收函数并返回函数的高阶函数。")
    run_with([
        lambda _m: _msg(tool_calls=[_tool_call("generate_quiz", '{"topic": "装饰器", "count": 1}')]),
        lambda _m: _msg(content=quiz_text),
        lambda _m: _msg(content=quiz_text),
    ])
    answer, log_lines = rag.agent("出几道题", chunks, bm25, [])
    ok &= check("出题结果不被误拦", answer == quiz_text, answer[:24])
    ok &= check("日志记录了出题步骤", any("出题「装饰器」" in x for x in log_lines))

    log()
    fails = sum(1 for line in OUT if line.startswith("FAIL"))
    log(f"结果：{fails} 项失败" if fails else "结果：全部通过")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
