#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
会话持久化 · 自动判卷

这是《会话持久化_任务卡》的配套判卷脚本。你只需要让它全绿，不用改这个文件。

用法：
    python 测试_会话持久化.py              # 跑全部
    python 测试_会话持久化.py --list       # 只看测试名
    python 测试_会话持久化.py --only 3     # 只跑第 3 条

它做什么：
    把 AGENT_DB 指到一个临时目录里的库，
    所以不会碰你的真实 会话.db —— 前提是你实现了 AGENT_DB（任务卡里的硬要求）。
    第 0 条测试就是确认这件事；如果不满足，测试会直接中止，保护你的数据。

成本：跑一遍约 16 次 API 调用，约 ¥0.2，2~3 分钟。
"""

import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------- 配置

HERE = Path(__file__).resolve().parent
PROJECT = next(
    (p for p in [HERE, *HERE.parents] if (p / "05-RAG-Agent项目").is_dir()),
    HERE.parent,
)
TOOL = PROJECT / "cli.py"
if not TOOL.exists():
    TOOL = PROJECT / "我的切块工具.py"
REAL_DB = PROJECT / "会话.db"
PYTHON = sys.executable

TMP_BASE = Path(os.environ.get("SESSION_TEST_TMP", tempfile.gettempdir())).resolve()
TMPDIR = TMP_BASE / f"agent-session-test-{os.getpid()}"
shutil.rmtree(TMPDIR, ignore_errors=True)
TMPDIR.mkdir(parents=True, exist_ok=True)
TEST_DB = TMPDIR / "test.db"

CALL_TIMEOUT = 240          # 单次调用超时（秒）
SESSIONS_MAX_SECONDS = 2.0  # --sessions 的耗时上限

# ---------------------------------------------------------------- 小工具


def dwidth(s):
    """显示宽度：中文/全角算 2 列，其余算 1 列。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def ensure_api_key():
    if os.environ.get("DEEPSEEK_API_KEY"):
        return True
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as k:
            v, _ = winreg.QueryValueEx(k, "DEEPSEEK_API_KEY")
            if v:
                os.environ["DEEPSEEK_API_KEY"] = v
                return True
    except Exception:
        pass
    return False


def run(args, stdin="", timeout=CALL_TIMEOUT):
    """跑一次工具。返回 (returncode, 输出, 耗时秒)。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["AGENT_DB"] = str(TEST_DB)

    t0 = time.time()
    try:
        p = subprocess.run(
            [PYTHON, str(TOOL), *args],
            cwd=str(PROJECT),
            env=env,
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return p.returncode, (p.stdout or "") + (p.stderr or ""), time.time() - t0
    except subprocess.TimeoutExpired:
        return -1, f"<超时 {timeout}s —— 大概率卡在 input() 等输入>", time.time() - t0
    except FileNotFoundError as e:
        return -2, f"<跑不起来：{e}>", time.time() - t0


def reset_db():
    """删掉临时库，让会话号从 1 开始（只动临时库，不动真实库）。"""
    for suffix in ("", "-journal", "-wal", "-shm"):
        p = Path(str(TEST_DB) + suffix)
        if p.exists() and p.is_file():
            p.unlink()


def db_query(sql, params=()):
    conn = sqlite3.connect(str(TEST_DB))
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def db_tables():
    try:
        return [r[0] for r in db_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    except sqlite3.Error:
        return []


def find_table(required_cols):
    """按字段名找表，不依赖你起的表名。"""
    for t in db_tables():
        try:
            cols = {r[1] for r in db_query(f"PRAGMA table_info({t})")}
        except sqlite3.Error:
            continue
        if required_cols <= cols:
            return t, cols
    return None, set()


def session_id_of(out):
    """从输出里抓会话号。"""
    m = re.search(r"会话\s*#\s*(\d+)", out)
    return m.group(1) if m else None


def sid_label(out):
    """给失败信息用的显示：没抓到就明说，别显示 #None 让人以为是会话号叫 None。"""
    s = session_id_of(out)
    return f"#{s}" if s else "输出里没有会话状态行"


def round_of(out):
    m = re.search(r"第\s*(\d+)\s*轮", out)
    return int(m.group(1)) if m else None


def after_round(out, n):
    """取"第 n 轮"标记之后的输出——用来确认某一轮自己答了什么。"""
    m = re.search(r"第\s*%d\s*轮" % n, out)
    return out[m.start():] if m else out


def count_decisions(out):
    """数执行轨迹里"模型决策"出现了几次（一次重试会多一次）。"""
    return len(re.findall(r"模型决策", out))


# ---------------------------------------------------------------- 判分框架

RESULTS = []
CURRENT = {"no": None, "title": ""}


def section(no, title):
    CURRENT["no"], CURRENT["title"] = no, title
    print(f"\n{'=' * 66}\n测试 {no} · {title}\n{'=' * 66}")


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
    section(0, "保护：确认 AGENT_DB 生效，不碰真实库")
    info(f"临时库：{TEST_DB}")
    info(f"真实库：{REAL_DB}")

    existed_before = REAL_DB.exists()
    mtime_before = REAL_DB.stat().st_mtime if existed_before else None

    reset_db()
    rc, out, _ = run(["装饰器是干什么的"])

    db_created = TEST_DB.exists()
    existed_after = REAL_DB.exists()
    mtime_after = REAL_DB.stat().st_mtime if existed_after else None
    touched = (not existed_before and existed_after) or (
        existed_before and mtime_after != mtime_before)

    check("临时库被创建", db_created, "工具没有读 AGENT_DB，写到了别的地方")
    check("真实库没被动过", not touched,
          "工具忽略了 AGENT_DB，改到了真实 会话.db —— 请先实现环境变量覆盖")

    if not (db_created and not touched):
        print("\n" + "!" * 66)
        print("中止：AGENT_DB 还没生效。继续跑会污染你的真实会话记录。")
        print("请先完成任务卡第三节的『路径要能被环境变量 AGENT_DB 覆盖』。")
        print("!" * 66)
        return False
    return True


# ---------------------------------------------------------------- 小步 2-1

def test_1_basic():
    section(1, "建库 + 落库（小步 2-1）")
    reset_db()
    rc, out, _ = run(["装饰器是干什么的"])

    check("程序正常退出", rc == 0, f"exit={rc}\n{out[-500:]}")
    check("打印了会话状态行", session_id_of(out) is not None,
          "没找到『会话 #N』——格式见任务卡第四节")
    check("显示第 1 轮", round_of(out) == 1, f"抓到的是 {round_of(out)}")

    sess_t, _ = find_table({"title", "created_at", "updated_at"})
    msg_t, _ = find_table({"role", "content", "session_id"})
    check("有两张表（会话表 / 消息表）", sess_t and msg_t,
          f"现有表：{db_tables()}")

    if not (sess_t and msg_t):
        return
    n_sess = db_query(f"SELECT COUNT(*) FROM {sess_t}")[0][0]
    n_msg = db_query(f"SELECT COUNT(*) FROM {msg_t}")[0][0]
    check("落了 1 个会话", n_sess == 1, f"实际 {n_sess}")
    check("落了 2 条消息（1 问 1 答）", n_msg == 2, f"实际 {n_msg}")

    roles = sorted(r[0] for r in db_query(f"SELECT DISTINCT role FROM {msg_t}"))
    check("role 是 user + assistant", roles == ["assistant", "user"], f"实际 {roles}")

    # 顺序：user 在前
    order = [r[0] for r in db_query(f"SELECT role FROM {msg_t} ORDER BY id")]
    check("顺序是 user 在前", order[:1] == ["user"], f"实际 {order}")


def test_2_new_session():
    section(2, "不带 -c 就是新会话（小步 2-1）")
    reset_db()
    _, out1, _ = run(["装饰器是干什么的"])
    _, out2, _ = run(["刷题按什么顺序安排"])

    s1, s2 = session_id_of(out1), session_id_of(out2)
    check("第一次有会话号", s1 is not None)
    check("第二次有会话号", s2 is not None)
    check("两次会话号不同（没被粘到上一个会话）", s1 != s2, f"都是 #{s1}")

    sess_t, _ = find_table({"title", "created_at"})
    if sess_t:
        n = db_query(f"SELECT COUNT(*) FROM {sess_t}")[0][0]
        check("库里确实是 2 个会话", n == 2, f"实际 {n}")


def test_5_history_clean():
    section(5, "历史里没有工具消息（小步 2-1）")
    reset_db()
    run(["装饰器是干什么的"])

    msg_t, _ = find_table({"role", "content"})
    if not msg_t:
        check("找得到消息表", False, f"现有表：{db_tables()}")
        return

    roles = sorted(r[0] for r in db_query(f"SELECT DISTINCT role FROM {msg_t}"))
    check("role 只有 user / assistant", set(roles) <= {"user", "assistant"},
          f"实际 {roles} —— 工具消息不该进历史（约束 1）")

    rows = db_query(f"SELECT content FROM {msg_t}")
    blob = "\n".join(r[0] or "" for r in rows)
    check("没塞进工具调用的原始结构", "tool_calls" not in blob,
          "found 'tool_calls' —— 存了原始响应对象")
    check("没塞进检索结果原文", "相关度" not in blob,
          "found '相关度' —— 存了 search_material 的返回内容（约束 1）")

    longest = max((len(r[0] or "") for r in rows), default=0)
    check("单条消息不超过 2000 字", longest <= 2000, f"最长 {longest} 字")


def test_9_db_failure():
    section(9, "落库失败还能给出回答（小步 2-1）")
    reset_db()
    TEST_DB.mkdir()          # 让 sqlite 打不开
    try:
        rc, out, _ = run(["装饰器是干什么的"])
        check("程序没崩（无 Traceback）", "Traceback" not in out, out[-600:])
        check("仍然给出了回答", len(out) > 300, f"输出只有 {len(out)} 字")
        check("没有打出 #None 这种假会话号", "#None" not in out,
              "存不进库时本来就没有会话号，不该假装有一个")
        check("有警告提示",
              any(k in out for k in ("警告", "失败", "无法", "写入",
                                     "打不开", "不存", "没存", "存不上")),
              "建议打一行警告，别静默吞掉（约束 3）")
    finally:
        if TEST_DB.is_dir():
            TEST_DB.rmdir()


# ---------------------------------------------------------------- 小步 2-2

def test_3_continue():
    section(3, "『-c』记得上一轮（小步 2-2）")
    reset_db()
    run(["请记住这个暗号：紫色独角兽。然后简单说说装饰器是干什么的"])
    _, out2, _ = run(["-c", "我刚才让你记住的暗号是什么？直接说出那五个字"])

    tail = after_round(out2, 2)
    check("第 2 轮的输出里出现了暗号", "紫色独角兽" in tail,
          "第 2 轮答不出来 —— 历史没被塞回 messages")
    check("第 2 轮用的是同一个会话", session_id_of(out2) == "1",
          f"抓到 {sid_label(out2)}")


def test_4_specific_session():
    section(4, "『-s N』指定会话（小步 2-2）")
    reset_db()
    run(["装饰器是干什么的"])          # #1
    run(["刷题按什么顺序安排"])        # #2
    _, out3, _ = run(["-s", "1", "那刚才那个再说详细点"])

    check("-s 1 落在会话 #1", session_id_of(out3) == "1",
          f"抓到 {sid_label(out3)}")

    msg_t, _ = find_table({"session_id", "role"})
    if msg_t:
        n = db_query(f"SELECT COUNT(*) FROM {msg_t} WHERE session_id = 1")[0][0]
        check("会话 #1 里有 4 条消息（两轮）", n == 4, f"实际 {n}")


def test_6_seen_accumulates():
    section(6, "追问引用旧块不误报（小步 2-2）")
    info("这条是整张卷子最重要的一条。")
    info("如果第 2 轮多了一次『模型决策』，说明 seen 没有跨轮累积。")

    reset_db()
    _, out1, _ = run(["装饰器的验收标准是什么"])
    _, out2, _ = run(["-c", "刚才那条验收标准，原样再念一遍，并标出块号出处"])

    n1, n2 = count_decisions(out1), count_decisions(out2)
    info(f"第 1 轮模型决策 {n1} 次 / 第 2 轮 {n2} 次")

    cited = re.search(r"块\s*#\s*(\d+)", out2) is not None
    if not cited:
        check("第 2 轮引用了块号（否则本测试无效）", False,
              "模型这次没标块号，重跑一次；连续两次说明提示词要更明确")
        return

    check("第 2 轮没有触发校验重试", n2 <= 2,
          f"第 2 轮出现了 {n2} 次模型决策，第 1 轮只有 {n1} 次。\n"
          f"       多出来的那次大概率是引用校验把旧块判成了『本次没检索过』，\n"
          f"       回灌了错误文本让模型重答。检查 seen 是不是每轮都从 {{}} 开始。")

    check("第 2 轮没有道歉/纠正的痕迹",
          not any(k in out2 for k in ("抱歉，我重新", "重新回答", "我引用了不存在的")),
          "输出里出现了『重新回答』之类的字样")


def test_12_round_count():
    section(12, "轮数计数正确（小步 2-2 / 2-3）")
    reset_db()
    run(["装饰器是干什么的"])
    run(["-c", "那闭包呢"])
    _, out3, _ = run(["-c", "那生成器呢"])

    check("第 3 次续聊显示『第 3 轮』", round_of(out3) == 3, f"抓到 {round_of(out3)}")
    check("仍然是同一个会话", session_id_of(out3) == "1",
          f"抓到 {sid_label(out3)}")


# ---------------------------------------------------------------- 小步 2-3

def test_7_sessions_list():
    section(7, "『--sessions』列表 + 不读知识库（小步 2-3）")
    reset_db()
    run(["装饰器是干什么的"])
    run(["刷题按什么顺序安排"])

    rc, out, dt = run(["--sessions"])
    info(f"耗时 {dt:.2f} 秒")

    check("正常退出", rc == 0, f"exit={rc}\n{out[-400:]}")
    listed = re.findall(r"#\s*(\d+)", out)
    check("列出了两个会话", len(set(listed)) >= 2, f"抓到 {listed}")
    if "1" in listed and "2" in listed:
        check("最近的排在前面", out.index("#2") < out.index("#1"),
              "#2 应该出现在 #1 上面（按 updated_at 倒序）")
    check("包含了轮数", "轮" in out)
    check("包含了标题", "装饰器" in out or "刷题" in out)

    check(f"耗时 < {SESSIONS_MAX_SECONDS} 秒（说明没读知识库）", dt < SESSIONS_MAX_SECONDS,
          f"实测 {dt:.2f}s —— main() 要在 load_all() 之前分流（约束 6）")


def test_11_alignment():
    section(11, "会话列表中文对齐（小步 2-3）")
    reset_db()
    run(["装饰器是干什么的"])
    run(["刷题按什么顺序安排"])
    _, out, _ = run(["--sessions"])

    lines = [l for l in out.splitlines()
             if l.strip().startswith("#") and re.match(r"#\s*\d+", l.strip())]
    if len(lines) < 2:
        check("至少抓到两行会话", False, f"抓到 {lines}")
        return

    widths = [dwidth(l) for l in lines]
    for l, w in zip(lines, widths):
        info(f"{w:>4} 列 | {l}")
    check("每行显示宽度一致", len(set(widths)) == 1,
          f"宽度集合 {sorted(set(widths))} —— 用 pad()/dwidth()，别用 :<16")


def test_8_no_db():
    section(8, "没有库也能跑（小步 2-3）")
    reset_db()
    rc, out, dt = run(["--sessions"])
    check("正常退出（不报错）", rc == 0, f"exit={rc}\n{out[-400:]}")
    check("没有 Traceback", "Traceback" not in out, out[-400:])
    check("有友好的空提示", ("没有" in out or "暂无" in out or "空" in out
                             or len(out.strip()) < 60),
          f"建议打印一句『还没有任何会话』，实际输出：{out.strip()[:200]}")


def test_10_interactive():
    section(10, "交互模式连续追问（小步 2-3）")
    reset_db()
    stdin = (
        "请记住这个暗号：蓝色河马。然后在回答里不要写出这三个字，简单说说装饰器。\n"
        "我刚才让你记住的暗号是什么？\n"
        "exit\n"
    )
    rc, out, dt = run([], stdin=stdin, timeout=400)
    info(f"耗时 {dt:.1f} 秒")

    check("正常退出", rc == 0, f"exit={rc}\n{out[-600:]}")
    check("没有超时（说明 exit 被识别了）", rc != -1)

    tail = after_round(out, 2)
    check("第 2 轮答出了暗号", "蓝色河马" in tail,
          "交互模式里历史没累积 —— 每轮要复用同一个 messages 列表")
    check("两轮在同一个会话里", session_id_of(tail) == "1",
          f"第 2 轮抓到 {sid_label(tail)}")

    msg_t, _ = find_table({"session_id", "role"})
    if msg_t:
        n = db_query(f"SELECT COUNT(*) FROM {msg_t}")[0][0]
        check("两轮都落库了（4 条消息）", n == 4, f"实际 {n}")


# ---------------------------------------------------------------- 主流程

ALL = [
    (0, "保护：确认 AGENT_DB 生效，不碰真实库", test_0_guard),
    (1, "建库 + 落库（小步 2-1）", test_1_basic),
    (2, "不带 -c 就是新会话（小步 2-1）", test_2_new_session),
    (5, "历史里没有工具消息（小步 2-1）", test_5_history_clean),
    (9, "落库失败还能给出回答（小步 2-1）", test_9_db_failure),
    (3, "『-c』记得上一轮（小步 2-2）", test_3_continue),
    (4, "『-s N』指定会话（小步 2-2）", test_4_specific_session),
    (6, "追问引用旧块不误报（小步 2-2）★核心", test_6_seen_accumulates),
    (12, "轮数计数正确（小步 2-2 / 2-3）", test_12_round_count),
    (7, "『--sessions』列表 + 不读知识库（小步 2-3）", test_7_sessions_list),
    (11, "会话列表中文对齐（小步 2-3）", test_11_alignment),
    (8, "没有库也能跑（小步 2-3）", test_8_no_db),
    (10, "交互模式连续追问（小步 2-3）", test_10_interactive),
]


def main():
    args = sys.argv[1:]

    if "--list" in args:
        print("可用测试：")
        for no, title, _fn in ALL:
            print(f"  {no:>2}  {title}")
        return 0

    only = None
    if "--only" in args:
        i = args.index("--only")
        if i + 1 < len(args):
            only = int(args[i + 1])

    if not TOOL.exists():
        print(f"✗ 找不到工具：{TOOL}")
        return 2
    if not ensure_api_key():
        print("✗ 没找到 DEEPSEEK_API_KEY（环境变量和注册表里都没有）")
        return 2

    print("会话持久化 · 自动判卷")
    print(f"工具  ：{TOOL}")
    print(f"临时库：{TEST_DB}")

    if not test_0_guard():
        return 2

    for no, title, fn in ALL:
        if no == 0:
            continue
        if only is not None and no != only:
            continue
        try:
            fn()
        except Exception as e:
            RESULTS.append((f"{no}.?", False, f"{type(e).__name__}: {e}"))
            print(f"  ❌ {no}.? 测试自己抛异常了：{type(e).__name__}: {e}")

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)

    print("\n" + "=" * 66)
    print(f"结果：{passed}/{total} 通过")
    print("=" * 66)
    for name, ok, _ in RESULTS:
        print(f"  {'✅' if ok else '❌'} {name}")

    if passed < total:
        print("\n失败的条目：")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  ❌ {name}")
                if detail:
                    print(f"     {detail}")
        return 1

    print("\n🎉 全绿。会话持久化完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
