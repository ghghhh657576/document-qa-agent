# -*- coding: utf-8 -*-
"""Web 入口：FastAPI 应用 + 页面/JSON 路由（引擎复用同目录 rag.py）。"""
from pathlib import Path
import sys
import time
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from fastapi import Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

# 检索/Agent 引擎就在本目录的 rag.py（命令行与 Web 共用同一份，不复制）
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import rag

templates = Jinja2Templates(directory=BASE_DIR / "templates")

CHUNKS = rag.load_all(rag.BASE / "data", 500)
BM25 = rag.build_index(CHUNKS)
CONN = rag.init_db()
def get_page_sessions():
    if CONN is None:
        return []
    with rag.DB_LOCK:
        rows = CONN.execute(
            "SELECT id, title, updated_at FROM session ORDER BY updated_at DESC, id DESC"
        ).fetchall()
    return [{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]

def trace_summary(trace, total_elapsed):
    """Web专用：把执行轨迹汇总成结构化数据，复用rag里的价格常量，不打印"""
    if not trace:
        return None
    rows = []
    for r in trace:
        rows.append({
            "step": r["step"],
            "name": r["name"],
            "kind": r["kind"],
            "prompt": r.get("prompt"),
            "completion": r.get("completion"),
            "cache_hit": r.get("cache_hit"),
            "elapsed": r["elapsed"],
        })
    n_model = sum(1 for r in trace if r["kind"] == "model")
    n_tool = sum(1 for r in trace if r["kind"] == "tool")
    p_in = sum(r.get("prompt", 0) for r in trace)
    p_out = sum(r.get("completion", 0) for r in trace)
    p_hit = sum(r.get("cache_hit", 0) for r in trace)
    cost = ((p_in - p_hit) * rag.PRICE_MISS + p_hit * rag.PRICE_HIT + p_out * rag.PRICE_OUT) / 1_000_000
    cache_rate = (p_hit / p_in * 100) if p_in else None
    return {
        "rows": rows,
        "total_elapsed": total_elapsed,
        "cache_rate": cache_rate,
        "cost": cost,
        "n_model": n_model,
        "n_tool": n_tool,
        "p_in": p_in,
        "p_out": p_out,
    }
print(f"[启动] 已加载 {len(CHUNKS)} 块，索引已建好")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html",
        context={"answer": None, "question": "", "sid": None,
                 "history": [], "sessions": get_page_sessions() if CONN is not None else []
        }
    )

@app.post("/delete-session")
def delete_session(sid: str = Form("")):
    # 数据库异常/参数非法直接跳回首页，不抛错误页
    try:
        with rag.DB_LOCK:
            if CONN is not None and sid.strip().isdigit():
                del_sid = int(sid.strip())
                exists = CONN.execute(
                    "SELECT 1 FROM session WHERE id = ?", (del_sid,)
                ).fetchone()
                if exists:
                    # 先删消息，再删会话，避免外键约束和孤儿数据
                    CONN.execute(
                        "DELETE FROM message WHERE session_id = ?", (del_sid,)
                    )
                    CONN.execute(
                        "DELETE FROM session WHERE id = ?", (del_sid,)
                    )
                    CONN.commit()
    except Exception as e:
        try:
            with rag.DB_LOCK:
                if CONN is not None:
                    CONN.rollback()
        except Exception:
            pass
        print(f"删除会话失败：{e}")
    # 无论成功失败都回到首页
    return RedirectResponse(url="/", status_code=302)

@app.get("/chat/{sid}")
def open_chat(request: Request, sid: int):
    # 数据库不可用时不要抛 500：直接回首页
    if CONN is None:
        return RedirectResponse(url="/", status_code=302)
    # 校验会话是否真实存在（和写操作共用一把锁）
    with rag.DB_LOCK:
        exists = CONN.execute("SELECT 1 FROM session WHERE id = ?", (sid,)).fetchone()
    if not exists:
        # 非法/已删除会话直接回到首页
        return RedirectResponse(url="/", status_code=302)
    history = rag.load_history(CONN, sid)
    return templates.TemplateResponse(
        request=request, name="index.html",
        context={
            "answer": None, "question": "", "sid": sid,
            "history": history, "sessions": get_page_sessions()
        }
    )

@app.post("/ask-page")
def ask_page(request: Request, question: str = Form(""), sid: str = Form("")):
    q = question.strip()
    # 解析并校验会话ID，非法/不存在的ID视为新会话
    current_sid = None
    if sid.strip().isdigit() and CONN is not None:
        tmp_sid = int(sid.strip())
        with rag.DB_LOCK:
            exists = CONN.execute("SELECT 1 FROM session WHERE id = ?", (tmp_sid,)).fetchone()
        if exists:
            current_sid = tmp_sid
    # 加载当前会话的全部历史
    history = []
    if current_sid is not None and CONN is not None:
        history = rag.load_history(CONN, current_sid)
    # 空问题校验，不创建新会话
    if not q:
        return templates.TemplateResponse(
            request=request, name="index.html",
            context={
                "answer": "问题不能为空", "question": question,
                "sid": current_sid, "history": history,
                "sessions": get_page_sessions() if CONN is not None else [],
                "detail": None, "log": None
            },
            status_code=400
        )
    # 服务端长度校验：不能只依赖前端 maxlength
    if len(q) > 2000:
        return templates.TemplateResponse(
            request=request, name="index.html",
            context={
                "answer": "问题太长，最多 2000 字", "question": question,
                "sid": current_sid, "history": history,
                "sessions": get_page_sessions() if CONN is not None else [],
                "detail": None, "log": None
            },
            status_code=400
        )
    # 首次有效提问，自动新建会话；失败时清理，避免留下空会话
    created_new = False
    if current_sid is None and CONN is not None:
        current_sid = rag.start_session(CONN, q)
        created_new = current_sid is not None
    # 调用Agent，传入历史上下文实现多轮理解，记录执行轨迹
    trace = []
    t0 = time.time()
    try:
        answer, log = rag.agent(q, CHUNKS, BM25, trace, history=history)
    except Exception as e:
        print(f"（网页端 Agent 异常：{e}）")
        if created_new:
            rag.drop_empty_session(CONN, current_sid)
        return templates.TemplateResponse(
            request=request, name="index.html",
            context={
                "answer": "服务出错了，请稍后再试", "question": question,
                "sid": current_sid, "history": history,
                "sessions": get_page_sessions() if CONN is not None else [],
                "detail": None, "log": None
            },
            status_code=500
        )
    total_elapsed = time.time() - t0
    detail = trace_summary(trace, total_elapsed)
    if answer is None:
        if created_new:
            rag.drop_empty_session(CONN, current_sid)
        return templates.TemplateResponse(
            request=request, name="index.html",
            context={
                "answer": "服务暂时不可用，请稍后再试", "question": question,
                "sid": current_sid, "history": history,
                "sessions": get_page_sessions() if CONN is not None else [],
                "detail": None, "log": None
            },
            status_code=503
        )
    # 本轮对话落库；保存失败时至少把本轮留在页面内存里
    if CONN is not None and current_sid is not None:
        saved = rag.save_turn(CONN, current_sid, q, answer)
        if saved:
            try:
                history = rag.load_history(CONN, current_sid)
            except Exception as e:
                print(f"（网页端历史读取失败：{e}）")
                history.append({"role": "user", "content": q})
                history.append({"role": "assistant", "content": answer})
        else:
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": answer})
            if log is not None:
                log.append("（提醒：本轮对话没有存进数据库，刷新页面后会丢失）")
            if created_new:
                rag.drop_empty_session(CONN, current_sid)
    else:
        # 数据库不可用时，本次请求内存保留上下文
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
    return templates.TemplateResponse(
        request=request, name="index.html",
        context={"answer": None, "question": "", "sid": current_sid,
                 "history": history, "sessions": get_page_sessions() if CONN is not None else [],
                 "detail": detail, "log": log
        }
    )

class Test(BaseModel):
    question:str= Field(..., max_length=2000)

@app.post("/ask")
def ask(t:Test):
    if not t.question.strip():
        raise HTTPException(400, "问题不能为空")
    answer, log = rag.agent(t.question, CHUNKS, BM25, [])
    if answer is None:
        raise HTTPException(503, "模型暂时不可用")
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
