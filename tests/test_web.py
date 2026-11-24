# -*- coding: utf-8 -*-
"""网页层冒烟测试：不依赖真实模型 Key，使用临时 SQLite 库。
运行（在仓库根目录）：
    python tests/test_web.py
可用 WEB_PROJECT 环境变量指定项目目录（默认仓库根目录，app.py 与 rag.py 同目录）。
"""
import os
import sys
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = Path(os.environ.get("WEB_PROJECT", ROOT)).resolve()
sys.path.insert(0, str(PROJECT))
os.environ["DEEPSEEK_API_KEY"] = ""
TMP_BASE = Path(os.environ.get("WEB_TEST_TMP", tempfile.gettempdir())).resolve()
TMP_DIR = TMP_BASE / f"web-test-{os.getpid()}"
shutil.rmtree(TMP_DIR, ignore_errors=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["AGENT_DB"] = str(TMP_DIR / "web_test.db")

from fastapi.testclient import TestClient  # noqa: E402
import app as web  # noqa: E402

client = TestClient(web.app, raise_server_exceptions=False)
passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


def clean():
    if web.CONN is None:
        return
    web.CONN.execute("DELETE FROM message")
    web.CONN.execute("DELETE FROM session")
    web.CONN.commit()


def counts():
    if web.CONN is None:
        return None
    return (
        web.CONN.execute("SELECT COUNT(*) FROM session").fetchone()[0],
        web.CONN.execute("SELECT COUNT(*) FROM message").fetchone()[0],
    )


def post_page(q, sid=""):
    return client.post("/ask-page", data={"question": q, "sid": sid})


def main():
    print(f"项目：{PROJECT}")
    if web.CONN is None:
        print("FAIL  数据库连接为空，无法测试")
        return 1

    # 1. 首页
    clean()
    r = client.get("/")
    check("GET / 200", r.status_code == 200, r.status_code)
    check("首页有提问表单", 'name="question"' in r.text)

    # 2. 空白问题：不建会话
    clean()
    r = post_page("   ")
    check("空白问题 400", r.status_code == 400, r.status_code)
    check("空白问题有提示", "问题不能为空" in r.text)
    check("空白问题不建会话", counts() == (0, 0), counts())

    # 3. 超长问题：服务端拦截，不建会话
    clean()
    r = post_page("啊" * 2500)
    check("超长问题 400", r.status_code == 400, r.status_code)
    check("超长问题有提示", "问题太长" in r.text)
    check("超长问题不建会话", counts() == (0, 0), counts())

    # 4. 无 Key：503 且不留下空会话
    clean()
    r = post_page("这是一次没有 Key 的测试")
    check("无 Key 返回 503", r.status_code == 503, r.status_code)
    check("无 Key 有提示", "服务暂时不可用" in r.text)
    check("无 Key 不留空会话", counts() == (0, 0), counts())

    # 5. Agent 异常：友好 500，不留空会话
    clean()
    orig_agent = web.rag.agent

    def boom(*args, **kwargs):
        raise RuntimeError("模拟 Agent 异常")

    web.rag.agent = boom
    try:
        r = post_page("触发异常")
        check("Agent 异常返回 500", r.status_code == 500, r.status_code)
        check("Agent 异常有友好提示", "服务出错了" in r.text)
        check("Agent 异常不留空会话", counts() == (0, 0), counts())
    finally:
        web.rag.agent = orig_agent

    # 6. 保存失败：答案仍在页面，且不留下空会话
    clean()
    orig_save = web.rag.save_turn
    web.rag.agent = lambda *a, **k: ("保存失败但页面要显示的答案", ["模拟日志"])
    web.rag.save_turn = lambda *a, **k: False
    try:
        r = post_page("保存失败测试")
        check("保存失败仍返回 200", r.status_code == 200, r.status_code)
        check("保存失败仍显示答案", "保存失败但页面要显示的答案" in r.text)
        check("保存失败不留空会话", counts() == (0, 0), counts())
    finally:
        web.rag.agent = orig_agent
        web.rag.save_turn = orig_save

    # 7. 正常保存 -> 详情页 -> 删除
    clean()
    web.rag.agent = lambda *a, **k: ("删除测试的答案", ["模拟日志"])
    try:
        r = post_page("删除测试的问题")
        check("正常提问 200", r.status_code == 200, r.status_code)
        check("正常提问显示答案", "删除测试的答案" in r.text)
        check("正常提问保存一轮", counts() == (1, 2), counts())
        sid = web.CONN.execute("SELECT id FROM session ORDER BY id DESC LIMIT 1").fetchone()[0]
        r = client.get(f"/chat/{sid}")
        check("会话详情 200", r.status_code == 200, r.status_code)
        r = client.post("/delete-session", data={"sid": str(sid)}, follow_redirects=False)
        check("删除会话 302", r.status_code == 302, r.status_code)
        check("删除后库为空", counts() == (0, 0), counts())
    finally:
        web.rag.agent = orig_agent

    # 8. JSON 接口的校验不变
    clean()
    r = client.post("/ask", json={"question": " "})
    check("/ask 空白 400", r.status_code == 400, r.status_code)
    r = client.post("/ask", json={"question": "啊" * 2500})
    check("/ask 超长 422", r.status_code == 422, r.status_code)

    # 9. /chat 对非法 ID 和数据库不可用都不 500
    clean()
    r = client.get("/chat/999999", follow_redirects=False)
    check("非法会话重定向", r.status_code == 302, r.status_code)
    orig_conn = web.CONN
    web.CONN = None
    try:
        r = client.get("/chat/1", follow_redirects=False)
        check("DB 不可用时 /chat 不 500", r.status_code == 302, r.status_code)
    finally:
        web.CONN = orig_conn

    if web.CONN is not None:
        web.CONN.close()
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(f"\n结果：{passed} 通过，{failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
