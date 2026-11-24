#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
接入 Agent · 小步 1-3 · 自动判卷

这是《接入Agent_任务卡》的配套判卷脚本。你只需要让它全绿，不用改这个文件。

用法：
    python 测试_接入Agent.py              # 跑全部
    python 测试_接入Agent.py --list       # 只看测试名
    python 测试_接入Agent.py --only 3     # 只跑第 3 条

它做什么：
    自己启动你的服务（端口 8139）→ 等它就绪 → 发 HTTP 请求 → 关掉服务。
    所以你不用手动先起服务，也不会和你开在 8000 的调试服务打架。

成本：约 ¥0.02（会真调两次模型）。
"""

import atexit
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------- 配置

HERE = Path(__file__).resolve().parent
PROJECT = next(
    (p for p in [HERE, *HERE.parents] if (p / "rag.py").is_file()),
    HERE.parent,
)
APP_DIR = PROJECT  # 单目录布局：Web 入口、引擎、模板都在仓库根目录
APP = APP_DIR / "app.py"
RAG = APP_DIR / "rag.py"
STOP = APP_DIR / "停用词表.txt"
DATA = APP_DIR / "data"
PYTHON = sys.executable

PORT = 8139
BASE = f"http://127.0.0.1:{PORT}"
BOOT_MAX = 120.0        # 建索引 + 导入 openai，给足时间
ASK_TIMEOUT = 150       # 一次问答最多等多久（agent 可能调 3 次模型）

STARTUP_RE = re.compile(r"\[启动\]\s*已加载\s*(\d+)\s*块")

TMPDIR = Path(tempfile.mkdtemp(prefix="agent_1_3_test_"))
LOG = TMPDIR / "server.log"

SERVER = None
LOGF = None
LOG_OK = True
READY_SECONDS = None

ANSWER_CACHE = {}


def prev_answer(question):
    """取缓存里某个问题的回答；没问过就返回空串。"""
    item = ANSWER_CACHE.get(question)
    if not item:
        return ""
    obj = item[4]
    return (obj or {}).get("answer") or "" if isinstance(obj, dict) else ""


# ---------------------------------------------------------------- 服务的起停


def start_server():
    global SERVER, LOGF, LOG_OK
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"          # 关键：不然 print 会被缓冲，日志里看不见

    try:
        LOGF = open(LOG, "w", encoding="utf-8", errors="replace")
        sink, LOG_OK = LOGF, True
    except OSError:
        LOGF, sink, LOG_OK = None, subprocess.DEVNULL, False

    SERVER = subprocess.Popen(
        [PYTHON, "-m", "uvicorn", "app:app", "--port", str(PORT)],
        cwd=str(APP_DIR),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=sink,
        stderr=subprocess.STDOUT,
    )
    return SERVER


def stop_server():
    global SERVER, LOGF
    if SERVER is not None and SERVER.poll() is None:
        SERVER.terminate()
        try:
            SERVER.wait(timeout=10)
        except subprocess.TimeoutExpired:
            SERVER.kill()
    SERVER = None
    if LOGF is not None:
        try:
            LOGF.close()
        except Exception:
            pass
        LOGF = None


atexit.register(stop_server)


def read_log():
    if LOGF is None and not LOG_OK:
        return ""
    try:
        LOGF.flush()
    except Exception:
        pass
    try:
        return LOG.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def server_log(tail=2000):
    if LOGF is None and not LOG_OK:
        return ("<这次没能保存服务日志（临时目录不可写）—— 请自己开一个终端跑 "
                f"`python -m uvicorn app:app --port {PORT}` 看报错>")
    txt = read_log().strip()
    return txt[-tail:] if txt else "<服务没有任何输出>"


def wait_ready():
    t0 = time.time()
    while time.time() - t0 < BOOT_MAX:
        if SERVER is not None and SERVER.poll() is not None:
            return False, time.time() - t0
        st, _body, _ct = http("GET", "/", timeout=5)
        if st > 0:
            return True, time.time() - t0
        time.sleep(0.5)
    return False, time.time() - t0


# ---------------------------------------------------------------- HTTP


def http(method, path, body=None, timeout=20):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace"), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), e.headers.get("Content-Type", "")
    except Exception as e:
        return -1, f"<连不上：{type(e).__name__}: {e}>", ""


def as_json(text):
    try:
        return json.loads(text)
    except Exception:
        return None


def is_4xx(st):
    return 400 <= st < 500


def ask(question):
    """带缓存的问答：同一条问题只花一次钱。返回 (状态码, 原文, ct, 秒数, 解析结果)。"""
    if question in ANSWER_CACHE:
        return ANSWER_CACHE[question]
    t0 = time.time()
    st, body, ct = http("POST", "/ask", {"question": question}, timeout=ASK_TIMEOUT)
    item = (st, body, ct, time.time() - t0, as_json(body))
    ANSWER_CACHE[question] = item
    return item


# ---------------------------------------------------------------- 判分框架

RESULTS = []
CURRENT = {"no": None, "title": ""}


def section(no, title):
    CURRENT["no"], CURRENT["title"] = no, title
    print(f"\n{'=' * 68}\n测试 {no} · {title}\n{'=' * 68}")


def check(name, ok, detail=""):
    RESULTS.append((f"{CURRENT['no']}.{name}", bool(ok), detail))
    print(f"  {'✅' if ok else '❌'} {CURRENT['no']}.{name}")
    if detail and not ok:
        print(f"       ↳ {detail}")
    return ok


def info(msg):
    print(f"     · {msg}")


# ---------------------------------------------------------------- 测试 0：保护


def test_0_guard():
    section(0, "保护：文件齐了、服务起来了")

    check("app.py 存在", APP.exists(), f"找不到 {APP}")
    check("rag.py 存在（引擎 + 命令行入口，Web 共用）", RAG.exists(),
          f"找不到 {RAG}\n应在仓库根目录：单引擎布局，不再复制副本")
    check("停用词表.txt 存在", STOP.exists(),
          f"找不到 {STOP}\n缺了不会报错，但检索会悄悄变差（任务卡坑 3）")

    txts = sorted(DATA.glob("*.txt")) if DATA.exists() else []
    check("data\\ 里至少 3 个 .txt", len(txts) >= 3,
          f"实际 {len(txts)} 个：{[p.name for p in txts]}\n"
          "准备那一节的第 1 条命令：Copy-Item \"..\\data\\*.txt\" \".\\data\\\" -Force")
    if txts:
        info(f"知识库：{', '.join(p.name for p in txts)}")

    if not (APP.exists() and RAG.exists() and STOP.exists() and len(txts) >= 3):
        return False

    alive = SERVER is not None and SERVER.poll() is None
    check("uvicorn 进程还活着", alive,
          f"进程已退出（returncode={SERVER.poll() if SERVER else 'N/A'}）\n"
          f"----- 服务输出 -----\n{server_log()}")

    check(f"{BOOT_MAX:.0f} 秒内能响应请求", READY_SECONDS is not None,
          f"超时了。最常见原因：导入 rag 时就报错（比如缩进/参数名）。\n"
          f"----- 服务输出 -----\n{server_log()}")

    if READY_SECONDS is not None:
        info(f"启动到可响应用了 {READY_SECONDS:.1f} 秒（含建索引）")
    return all(ok for _, ok, _ in RESULTS)


# ---------------------------------------------------------------- 测试 1：启动时就建好索引


def test_1_build_at_startup():
    section(1, "★ 索引在启动时建好（还没发过任何 /ask）")

    log = read_log()
    hits = STARTUP_RE.findall(log)
    info(f"日志里匹配到 {len(hits)} 次「[启动] 已加载 N 块」：{hits}")

    check("收到第一个请求之前，日志里已经有启动行", len(hits) >= 1,
          "没找到 `[启动] 已加载 N 块，索引已建好`。\n"
          "两种可能：① 那三行写在函数里了（要在 app = FastAPI() 下面、函数外面）\n"
          "          ② print 格式改了（判卷按原格式抓）\n"
          f"----- 服务输出前 1200 字 -----\n{server_log(1200)}")

    n = int(hits[0]) if hits else 0
    check("块数合理（≥ 50）", n >= 50,
          f"实际 {n} 块。太少说明 data\\ 没拷全，或者 load_all 的目录不对")

    check("启动行只有一次（导入模块只执行一次）", len(hits) == 1,
          f"出现了 {len(hits)} 次。检查是不是把这三行写在了某个会被调用多次的地方")


# ---------------------------------------------------------------- 测试 2：真回答


def test_2_real_answer():
    section(2, "★ POST /ask 真的回答（不再是 66）")
    st, body, ct, dt, obj = ask("装饰器的验收标准是什么")
    info(f"HTTP {st}  Content-Type: {ct or '(空)'}  耗时 {dt:.1f}s")
    info(f"body 前 300 字：{body.strip()[:300]!r}")

    check("状态码 200", st == 200, f"实际 {st}，body：{body.strip()[:300]}")
    check("Content-Type 是 JSON", "json" in (ct or "").lower(), f"实际 {ct!r}")

    if not isinstance(obj, dict):
        check("响应是 JSON 对象", False, f"实际 {body.strip()[:200]}")
        return

    ans = obj.get("answer")
    check("有 answer 字段", "answer" in obj, f"实际字段：{sorted(obj)}")
    ok_str = isinstance(ans, str) and ans.strip() != ""
    check("answer 是非空字符串", ok_str, f"实际 {type(ans).__name__}: {ans!r}")
    if not ok_str:
        return

    check("answer 不是占位内容（不是 66 / 不是 null）", ans.strip() not in {"66", "null", "None"},
          f"实际 {ans[:80]!r} —— 是不是还没把 return 66 换掉？")

    check("真的提到装饰器", "装饰器" in ans, f"回答前 200 字：{ans[:200]!r}")

    weird = ("暂时不可用", "模型暂时", "数据库打不开")
    check("不是降级后的错误提示", not any(w in ans for w in weird),
          f"回答里出现了降级提示，说明链路中间断了：{ans[:200]!r}")

    info(f"回答长度 {len(ans)} 字")


def test_3_citation():
    section(3, "★ 回答里带引用标记（引用校验链路是通的）")
    st, body, ct, dt, obj = ask("装饰器的验收标准是什么")
    ans = (obj or {}).get("answer") or ""
    cites = re.findall(r"块\s*#\s*(\d+)", ans)
    info(f"抓到引用：{cites[:8]}{' ...' if len(cites) > 8 else ''}")

    check("有引用标记「块 #N」", bool(cites),
          "回答里没有任何块号。可能：① rag.agent 返回的 content 被截断了\n"
          "② 你把 SYSTEM_PROMPT 改了（要求模型标注出处那段不能动）")
    check("引用不是编的（块号在合理范围内）",
          bool(cites) and all(0 <= int(c) < 500 for c in cites),
          f"实际 {cites}")


# ---------------------------------------------------------------- 测试 4：换一个问题也能答


def test_4_second_question():
    section(4, "换一个问题也能答（不是写死的一句话）")
    st, body, ct, dt, obj = ask("二分查找的边界怎么处理")
    info(f"HTTP {st}  耗时 {dt:.1f}s")
    ans = (obj or {}).get("answer") or ""
    info(f"回答前 160 字：{ans[:160]!r}")

    check("状态码 200", st == 200, f"实际 {st}")
    check("answer 非空", bool(ans.strip()), "空回答")
    check("内容跟二分查找有关", ("二分" in ans) or ("查找" in ans) or ("边界" in ans),
          f"回答前 200 字：{ans[:200]!r}")
    check("和第一个问题不是同一段话",
          bool(ans.strip()) and ans.strip()[:40] != prev_answer("装饰器的验收标准是什么").strip()[:40],
          "两个问题返回了同样的开头 —— 是不是把回答写死了？")


# ---------------------------------------------------------------- 测试 5：索引没被重建


def test_5_build_once():
    section(5, "★ 发完 2 个问题后，索引还是只建了 1 次")
    log = read_log()
    hits = STARTUP_RE.findall(log)
    info(f"现在日志里匹配到 {len(hits)} 次启动行")

    check("启动行仍然是 1 次（没有每请求重建索引）", len(hits) == 1,
          f"出现了 {len(hits)} 次。说明 load_all/build_index 被放进了请求处理函数里，\n"
          "每来一个请求就重建一次。任务卡第 2 处：那三行要在函数外面。")


# ---------------------------------------------------------------- 测试 6：校验没被改坏


def test_6_validation():
    section(6, "输入校验还生效（1-2 做过的不能丢）")
    cases = [
        ("空字符串", {"question": ""}),
        ("三个空格", {"question": "   "}),
        ("2500 字", {"question": "啊" * 2500}),
        ("缺 question 字段", {}),
    ]
    for label, payload in cases:
        st, body, ct = http("POST", "/ask", payload, timeout=30)
        info(f"{label}: HTTP {st}  {body.strip()[:100]!r}")
        check(f"{label} → 4xx", is_4xx(st),
              f"实际 {st}。500 说明程序崩了（用户输入不对是 4xx，程序崩了才是 5xx）")


def test_7_no_leak():
    section(7, "出错时不泄漏内部细节")
    st, body, ct = http("POST", "/ask", {"question": ""}, timeout=30)
    check("错误响应不是空的", bool(body.strip()), f"HTTP {st} 返回了空 body")
    check("没有 Traceback", "Traceback" not in body, f"body：{body[:300]!r}")
    check("没有泄漏源码路径", ".py" not in body, f"body：{body[:300]!r}")


# ---------------------------------------------------------------- 测试 8：GET / 回归


def test_8_root():
    section(8, "GET / 还能用（1-1 做过的不能丢）")
    st, body, ct = http("GET", "/")
    info(f"HTTP {st}  body：{body.strip()[:120]!r}")
    check("状态码 200", st == 200, f"实际 {st}")
    check("body 不是空的", bool(body.strip()), "空内容")


# ---------------------------------------------------------------- 主流程

ALL = [
    (0, "保护：文件齐了、服务起来了", test_0_guard),
    (1, "★ 索引在启动时建好", test_1_build_at_startup),
    (2, "★ /ask 真的回答", test_2_real_answer),
    (3, "★ 回答带引用", test_3_citation),
    (4, "换一个问题也能答", test_4_second_question),
    (5, "★ 索引只建一次", test_5_build_once),
    (6, "输入校验还生效", test_6_validation),
    (7, "不泄漏内部细节", test_7_no_leak),
    (8, "GET / 回归", test_8_root),
]


def main():
    global READY_SECONDS
    argv = sys.argv[1:]

    if "--list" in argv:
        print("可单独跑的测试：")
        for no, title, _fn in ALL:
            print(f"  {no}  {title}")
        return 0

    only = None
    if "--only" in argv:
        i = argv.index("--only")
        try:
            only = int(argv[i + 1])
        except (IndexError, ValueError):
            print("--only 后面要跟一个数字，比如 --only 3")
            return 2

    print("=" * 68)
    print("接入 Agent · 自动判卷（小步 1-3）")
    print("=" * 68)
    print(f"服务目录：{APP_DIR}")
    if APP.exists():
        mt = datetime.fromtimestamp(APP.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"被测文件：{APP}")
        print(f"          最后保存于 {mt} · {APP.stat().st_size} 字节")
    print(f"启动端口：{PORT}（不占你的 8000）")
    print(f"提示：会真调 2 次模型，约 ¥0.02，耐心等")

    start_server()
    ok, dt = wait_ready()
    READY_SECONDS = dt if ok else None

    todo = [t for t in ALL if only is None or t[0] == only]
    if only is not None and not todo:
        print(f"\n没有编号 {only} 的测试。用 --list 看看有哪些。")
        stop_server()
        return 2

    for no, title, fn in todo:
        try:
            fn()
        except Exception as e:
            section(no, title)
            check(f"测试 {no} 自身没崩", False, f"{type(e).__name__}: {e}")

    stop_server()

    print("\n" + "=" * 68)
    total = len(RESULTS)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"结果：{passed}/{total} 通过")
    print("=" * 68)

    for name, ok, _ in RESULTS:
        print(f"  {'✅' if ok else '❌'} {name}")

    failed = [(n, d) for n, ok, d in RESULTS if not ok]
    if failed:
        print("\n失败的条目：")
        for n, d in failed:
            print(f"  ❌ {n}")
            if d:
                for line in str(d).splitlines():
                    print(f"       {line}")

    print()
    if not failed:
        print("🎉 全绿。小步 1-3 完成。")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
