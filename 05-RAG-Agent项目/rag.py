import sys
import jieba
from rank_bm25 import BM25Okapi
import os
from openai import OpenAI
import json
import time
from pathlib import  Path
import re
import unicodedata
import sqlite3
from datetime import datetime

BASE = Path(__file__).resolve().parent

ALLOWED_TOOLS = {"search_material", "generate_quiz"}

#把正常输出和错误输出编码全部变为utf-8
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

#读取停用词表文件，消除常见词对评分干扰
STOP = set()
try:
    with open(BASE / "停用词表.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                STOP.add(line)
except FileNotFoundError:
    pass

#大模型可以调用的工具(搜索和出题)
TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_material",
        "description": "在课程资料中搜索相关内容。当你需要查找资料里的事实性信息时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                }
            },
            "required": ["query"]
        }
    }
},{
    "type": "function",
    "function": {
        "name": "generate_quiz",
        "description": "根据课程资料生成练习题。当用户想测试自己、要练习题、或要求出题时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "出题的知识点，例如「装饰器」「双指针」"
                },
                "count": {
                    "type": "integer",
                    "description": "题目数量，默认 3，最多 10"
                }
            },
            "required": ["topic"]
        }
    }
}]

#大模型提示词
SYSTEM_PROMPT = (
    "你是课程资料问答助手。只能依据 search_material 检索到的资料内容回答，"
    "回答时标注引用的块编号和它来自的资料文件名，格式如「（块 #12，知识点细化手册_clean.txt）」；"
    "资料中没有依据时直接说“资料中没有找到”，禁止编造。"
    "无论资料里有多少英文术语，始终用简体中文回答。"
)

#读取文件
def read_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="gb18030", errors="replace") as f:
                return f.read()
        except OSError as e:
            print(f"（读不了 {file_path}：{e}）")
            return ""
    except OSError as e:
        print(f"（读不了 {file_path}：{e}）")
        return ""

#把文章切块，尽量在一句话的结尾截断
def split_every(s, n):
    t=[]
    i=0
    if n>0:
        while len(s) - i >= n:
            if s[i+n-1] in "。！？\n":
                t.append(s[i:i+n])
                i+=n
            else:
                end=i+n
                while end < len(s) and s[end-1] not in "。！？\n" :
                    end+=1
                    if end-i-n==100:
                        end=i+n
                        break
                t.append(s[i:end])
                i=end
        if i<len(s):
            t.append(s[i:])
        return t
    else:
        return None

#分词函数
def tokenize(text):
    if not text or not text.strip():
        return []
    words = jieba.lcut(text)
    return [w for w in words if w.strip() and w not in STOP]

#读取data下所有txt,md文件，合并后分别读取并切片，包装成字典(带文件名)返回
def load_all(folder="data", n=500):
    if not Path(folder).is_dir():
        print(f"资料目录不存在：{folder}")
        return []
    txt_path=list(Path(folder).glob("*.txt"))
    md_path=list(Path(folder).glob("*.md"))
    all_path=sorted(txt_path+md_path)
    result=[]
    for f in all_path:
        text=read_file(f)
        chunk_list=split_every(text,n)
        if not chunk_list:
            continue
        chunks=[{"source":str(f.relative_to(folder)),"content":c}for c in chunk_list]
        result.extend(chunks)
    return result

#建索引
def build_index(chunks):
    docs = [tokenize(c["content"]) for c in chunks]
    if not any(docs):
        return None
    return BM25Okapi(docs)

#切分问题然后给字段打分，排序取前三位
def search(query, chunks, bm25, topk=3):
    qtokens = tokenize(query)
    scores = bm25.get_scores(qtokens)
    pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top = [(idx, sc) for idx, sc in pairs[:topk] if sc > 0]
    if not top:
        qset = set(qtokens)
        overlap = []
        for idx, ch in enumerate(chunks):
            hit = qset & set(tokenize(ch["content"]))
            if hit:
                overlap.append((len(hit), idx))
        overlap.sort(key=lambda x: (-x[0], x[1]))
        top = [(idx, float(cnt)) for cnt, idx in overlap[:topk]]
    if not top:
        return f"未找到与「{query}」相关的内容", []
    parts = [
        f"块 #{idx}（来自 {chunks[idx]['source']}，相关度 {sc:.2f}）：\n{chunks[idx]['content']}"
        for idx, sc in top
    ]
    return "\n\n".join(parts), top

CITE_RE = re.compile(r"块\s*#\s*(\d+)")

#找出返回结果所有的"块 #N"，去重后按出现顺序返回
def extract_citations(text):
    out = []
    for m in CITE_RE.finditer(text or ""):
        n = int(m.group(1))
        if n not in out:
            out.append(n)
    return out

#引用校验
def check_citations(answer, seen, total):
    answer = answer or ""
    cited = extract_citations(answer)
    if not cited:
        return "引用校验：回答里没有块引用（可能是拒答或纯寒暄）", []
    invalid = []
    lines = []
    for n in cited:
        if not (0 <= n < total):
            lines.append(f" 错误！ 块 #{n}：知识库里不存在（共 {total} 块，编号 0~{total - 1}）")
            invalid.append(n)
        elif n not in seen:
            lines.append(f" 错误！ 块 #{n}：存在，但本次检索从未返回过它")
            invalid.append(n)
        else:
            lines.append(f" 正确！ 块 #{n}（{seen[n]}）")
    head = f"引用校验：{len(cited)} 处引用，{len(invalid)} 处无效"
    return "\n".join([head] + lines), invalid

#从历史记录中提取被引用的块
def seed_seen(history, chunks):
    seen = {}
    for m in history:
        if m["role"] != "assistant":
            continue
        for num in CITE_RE.findall(m["content"]):
            i = int(num)
            if 0<=i<=len(chunks) - 1:
                seen[i] = chunks[i]["source"]
    return seen

#生成练习题
def do_generate_quiz(topic, count, chunks, bm25, client, trace):
    t0 = time.time()
    result, hits=search(topic, chunks, bm25)
    trace.append({
        "step": None, "kind": "tool", "name": "检索(出题)",
        "elapsed": time.time() - t0,
    })
    if not hits:
        return f"资料中没有关于「{topic}」的内容，无法出题。",[]
    prompt = (
        f"请根据下面的资料，出 {count} 道关于「{topic}」的练习题。\n"
        f"要求：\n"
        f"1. 每道题包含：题目、答案、一句话解析\n"
        f"2. 只依据资料内容，不要编造\n"
        f"3. 题目要有区分度，不要都是同一个考法\n\n"
        f"【资料】\n{result}"
    )
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是出题老师，只依据给定资料出题。"},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:
        trace.append({
            "step": None, "kind": "tool", "name": "生成题目失败",
            "elapsed": time.time() - t0,
        })
        return f"出题失败：{e}。可以稍后重试，或换一个更具体的知识点。", hits
    elapsed = time.time() - t0
    u = resp.usage
    trace.append({
        "step": None, "kind": "model", "name": "生成题目",
        "prompt": u.prompt_tokens,
        "completion": u.completion_tokens,
        "cache_hit": getattr(u, "prompt_cache_hit_tokens", 0),
        "elapsed": elapsed,
    })
    return resp.choices[0].message.content, hits

#主逻辑
def agent(question, chunks, bm25, trace, max_steps=5,history=None):
    if history is None:
        history = []
    if not question or not question.strip():
        return "请先输入一个问题。", []
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("没有找到 API Key（环境变量 DEEPSEEK_API_KEY）")
        return None, []
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    if bm25 is None:
        return "当前资料里没有可检索的内容（全是停用词或空白），请换一份资料或检查文件内容。", []
    log = []
    seen = seed_seen(history, chunks)
    citation_retried = False
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]+history+[{"role": "user", "content": question}]
    for step in range(max_steps):
        resp = None
        for attempt in range(3):
            try:
                t0 = time.time()
                resp = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    tools=TOOLS,
                )
                model_elapsed = time.time() - t0
                break
            except Exception as e:
                if attempt == 2:
                    print(f"接口连续 3 次调用失败：{e}")
                    return None, log
                time.sleep(2 ** attempt)
        u = resp.usage
        trace.append({
            "step": step + 1,
            "kind": "model",
            "name": "模型决策",
            "prompt": getattr(u, "prompt_tokens", 0),
            "completion": getattr(u, "completion_tokens", 0),
            "cache_hit": getattr(u, "prompt_cache_hit_tokens", 0),
            "elapsed": model_elapsed,
        })
        msg = resp.choices[0].message
        if not msg.tool_calls:
            report, invalid = check_citations(msg.content, seen, len(chunks))
            log.append(report)
            if invalid and not citation_retried:
                citation_retried = True
                log.append(f"引用校验失败：{invalid} 不是本次检索返回的块，已要求模型修正")
                messages.append(msg.model_dump())
                messages.append({
                    "role": "user",
                    "content": (
                        f"你刚才的回答引用了 {invalid}，但本次检索只返回过这些块："
                        f"{sorted(seen)}。请重新回答，只引用真实存在的块号。"
                    ),
                })
                continue
            if invalid:
                log.append(f"引用校验仍失败：{invalid}（已重试过一次）")
            return msg.content, log
        messages.append(msg.model_dump())
        for j,tc in enumerate(msg.tool_calls, start=1):
            if tc.function.name not in ALLOWED_TOOLS:
                result = f"错误：不存在名为「{tc.function.name}」的工具。可用工具：{', '.join(ALLOWED_TOOLS)}"
                log.append(f"第 {step + 1} 步-{j}：拒绝未知工具「{tc.function.name}」")
            else:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                if name == "search_material":
                    q = args.get("query")
                    q = q.strip() if isinstance(q, str) else ""
                    if not q:
                        result = "错误：搜索关键词为空，请提供 2~5 个字的核心术语"
                        log.append(f"第 {step + 1} 步-{j}：拒绝空关键词")
                    else:
                        t0 = time.time()
                        result, hits = search(q, chunks, bm25)
                        for i, sc in hits:
                            seen[i] = chunks[i]["source"]
                        trace.append({
                            "step": step + 1,
                            "kind": "tool",
                            "name": "search_material",
                            "elapsed": time.time() - t0,
                        })
                        log.append(f"第 {step + 1} 步-{j}：搜索「{q}」→ 返回 {len(hits)} 块")
                elif name == "generate_quiz":
                    topic = args.get("topic")
                    topic = topic.strip() if isinstance(topic, str) else ""
                    if not topic:
                        result = "错误：出题知识点为空，请提供具体的知识点名称"
                        log.append(f"第 {step + 1} 步-{j}：拒绝空知识点")
                    else:
                        count = args.get("count", 3)
                        if isinstance(count, bool):
                            count = 3
                        try:
                            count = int(count)
                        except (TypeError, ValueError):
                            count = 3
                        count = max(1, min(count, 10))
                        result, hits = do_generate_quiz(topic, count, chunks, bm25, client, trace)
                        for i, sc in hits:
                            seen[i] = chunks[i]["source"]
                        for rec in trace:
                            if rec["step"] is None:
                                rec["step"] = step + 1
                        log.append(f"第 {step + 1} 步-{j}：出题「{topic}」{count} 道")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    return "达到最大步数限制，未能完成", log

PRICE_MISS = 2.0
PRICE_HIT  = 0.5
PRICE_OUT  = 8.0
DB_PATH = Path(os.environ.get("AGENT_DB") or BASE / "会话.db")

#宽字符长度
def dwidth(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))

#填充
def pad(s, n):
    s = str(s)
    return s + " " * max(0, n - dwidth(s))

#截断
def cut(s, n):
    s = str(s)
    out, w = [], 0
    for ch in s:
        cw = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if w + cw > n:
            break
        out.append(ch)
        w += cw
    return "".join(out)

#打印执行轨迹
def print_trace(trace, total_elapsed):
    if not trace:
        print("（没有记录）")
        return
    W_STEP, W_NAME, W_NUM, W_TIME, W_CACHE = 4, 18, 8, 8, 14
    print()
    print("执行轨迹")
    print("-" * 60)
    print(pad("步", W_STEP) + pad("用途", W_NAME) + pad("输入", W_NUM)
          + pad("输出", W_NUM) + pad("耗时", W_TIME) + pad("缓存", W_CACHE))
    for r in trace:
        line = pad(r["step"], W_STEP) + pad(cut(r["name"], W_NAME - 2), W_NAME)
        if r["kind"] == "model":
            line += (pad(r["prompt"], W_NUM) + pad(r["completion"], W_NUM)
                     + pad(f"{r['elapsed']:.1f}s", W_TIME)
                     + pad(f"{r['cache_hit']}/{r['prompt']}", W_CACHE))
        else:
            line += (pad("—", W_NUM) + pad("—", W_NUM)
                     + pad(f"{r['elapsed']:.1f}s", W_TIME) + pad("—", W_CACHE))
        print(line)
    print("-" * 60)
    n_model = sum(1 for r in trace if r["kind"] == "model")
    n_tool = sum(1 for r in trace if r["kind"] == "tool")
    p_in = sum(r.get("prompt", 0) for r in trace)
    p_out = sum(r.get("completion", 0) for r in trace)
    p_hit = sum(r.get("cache_hit", 0) for r in trace)
    cost = ((p_in - p_hit) * PRICE_MISS + p_hit * PRICE_HIT + p_out * PRICE_OUT) / 1_000_000
    print(f"合计：模型 {n_model} 次 / 工具 {n_tool} 次    "
          f"输入 {p_in}  输出 {p_out}   总耗时 {total_elapsed:.1f}s")
    if p_in:
        print(f"缓存命中率：{p_hit / p_in * 100:.0f}%")
    print(f"本次会话预估成本：¥{cost:.4f}")

#统一打印一轮问答的执行过程、答案与账单（main 和 interactive_loop 共用）
def render_round(answer, log, trace, elapsed):
    print("=" * 50)
    print("Agent 执行过程：")
    if not log:
        print("  （模型没有调用工具，直接回答）")
    for item in log:
        print("  " + item)
    print()
    print("=" * 50)
    print("回答：")
    print(answer)
    print_trace(trace, elapsed)

#失败退出：打印原因，若当前会话是刚新建的则清理掉空会话
def fail_exit(conn, sid, new_session, msg):
    print(msg)
    if new_session:
        drop_empty_session(conn, sid)

#初始化数据库
def init_db():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript("""
CREATE TABLE IF NOT EXISTS session (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    updated_at TEXT,
    title TEXT
);
CREATE TABLE IF NOT EXISTS message (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES session(id)
);
CREATE INDEX IF NOT EXISTS idx_message_session ON message(session_id);
        """)
        conn.commit()
        return conn
    except Exception as e:
        print(f"（数据库打不开，这次不存对话：{e}）")
        return None

#开始会话
def start_session(conn, title):
    try:
        现在 = datetime.now().isoformat(timespec="seconds")
        cur = conn.execute(
            "INSERT INTO session (created_at, updated_at, title) VALUES (?, ?, ?)",
            (现在, 现在, title),
        )
        conn.commit()
        return cur.lastrowid
    except Exception as e:
        print(f"（会话没建成，这次不存对话：{e}）")
        return None

#删除空会话
def drop_empty_session(conn, sid):
    if conn is None or sid is None:
        return
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM message WHERE session_id = ?", (sid,)
        ).fetchone()[0]
        if n == 0:
            conn.execute("DELETE FROM session WHERE id = ?", (sid,))
            conn.commit()
    except Exception:
        pass

#保存轮次
def save_turn(conn, sid, question, answer):
    try:
        现在 = datetime.now().isoformat(timespec="seconds")
        conn.execute(
            "INSERT INTO message (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (sid, "user", question, 现在),
        )
        conn.execute(
            "INSERT INTO message (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (sid, "assistant", answer, 现在),
        )
        conn.execute("UPDATE session SET updated_at = ? WHERE id = ?", (现在, sid))
        conn.commit()
        return True
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        print(f"（这轮对话没存上：{e}）")
        return False

#加载历史记录
def load_history(conn, sid):
    sql=("""
SELECT role, content FROM message
WHERE session_id = ? AND role IN ('user', 'assistant')
ORDER BY id
    """)
    rows = conn.execute(sql, (sid,)).fetchall()
    history=[]
    for role,content in rows:
        history.append({"role":role,"content":content})
    return history

#列出会话
def list_sessions(conn):
    if conn is None:
        print("（数据库打不开，看不了会话列表）")
        return
    rows = conn.execute(
        "SELECT id, updated_at, title FROM session ORDER BY updated_at DESC, id DESC"
    ).fetchall()
    if not rows:
        print("还没有任何会话")
        return
    for sid, updated, title in rows:
        n = conn.execute(
            "SELECT COUNT(*) FROM message WHERE session_id = ?", (sid,)
        ).fetchone()[0]
        print(pad(f"#{sid}", 6)
              + pad(str(updated or "").replace("T", " "), 20)
              + pad(f"{n // 2} 轮", 8)
              + cut(title or "", 24))

#交互循环
def interactive_loop(t, bm25, conn, sid, history):
    round_no = len(history) // 2 + 1
    while True:
        try:
            q = input("你问（quit 退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue
        trace = []
        t_start = time.time()
        answer, log = agent(q, t, bm25, trace, history=history)
        elapsed_all = time.time() - t_start
        if answer is None:
            continue
        if sid is not None:
            print(f"（会话 #{sid}，第 {round_no} 轮）")
        elif conn:
            sid = start_session(conn, q)
            if sid is not None:
                print(f"（会话 #{sid}，第 {round_no} 轮）")
        render_round(answer, log, trace, elapsed_all)
        if conn and sid is not None:
            save_turn(conn, sid, q, answer)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})
        round_no += 1

#解析 -s / -c 标志，返回 (模式, 指定会话号, 剩余参数)；模式为 "s" / "c" / None
def parse_continue_flags(args):
    if "-s" in args:
        pos = args.index("-s")
        if pos + 1 < len(args) and args[pos + 1].isdigit():
            given_sid = int(args[pos + 1])
            del args[pos:pos + 2]
            return "s", given_sid, args
        del args[pos:pos + 1]
        print("（-s 后面要跟会话号，这次当普通提问）")
        return None, None, args
    if "-c" in args:
        args.remove("-c")
        return "c", None, args
    return None, None, args

#解析块大小与问题，返回 (n, query, n_only)
def parse_chunk_args(rest):
    n, query, n_only = 500, None, False
    if len(rest) >= 1:
        if rest[0].lstrip("-").isdigit():
            n = int(rest[0])
            if len(rest) >= 2:
                query = rest[1]
            else:
                n_only = True
        else:
            query = rest[0]
    return n, query, n_only

#把会话标志翻译成真实会话号：-s 用指定号，-c 查最近会话
def resolve_sid(conn, mode, given_sid):
    if conn is None or mode is None:
        return None
    if mode == "s":
        return given_sid
    row = conn.execute(
        "SELECT id FROM session ORDER BY updated_at DESC, id DESC LIMIT 1"
    ).fetchone()
    if row is not None:
        return row[0]
    return None

#判定资料来源：首参数是文件则读取，否则使用 data 目录；返回 None 表示应退出
def resolve_source(args):
    if args and os.path.isfile(args[0]):
        path = args[0]
        try:
            m = read_file(path)
        except FileNotFoundError:
            print(f"文件不存在：{path}")
            return None
        if not m.strip():
            print("文件内容为空，没有东西可切")
            return None
        return path, m, os.path.basename(path), args[1:]
    #参数看起来是资料文件名却找不到：明确报错，避免被误当成提问
    if args and args[0].lower().endswith((".txt", ".md")):
        print(f"文件不存在：{args[0]}，请检查文件名或路径（也可以把文件直接拖进终端）")
        return None
    return str(BASE / "data"), None, None, args

#加载并切块：目录走 load_all，单文件走 split_every，输出结构统一
def load_chunks(m, n, source_name):
    if m is None:
        return load_all(BASE / "data", n)
    return [{"source": source_name, "content": c} for c in split_every(m, n)]

#切块/分词演示：打印前 3 块原文与分词，不调用模型
def show_demo(chunks):
    for i, blk in enumerate(chunks[:3], start=1):
        text = blk["content"]
        words = list(jieba.cut(text))
        print(f"分词示例(第 {i} 块，来自 {blk['source']}，共 {len(words)} 个词)")
        print(" / ".join(words[:30]))
        print(f"--- 第 {i} 块 ---")
        print(text)
        print()

def main():
    args = sys.argv[1:]
    if "--sessions" in args:
        list_sessions(init_db())
        return
    mode, given_sid, args = parse_continue_flags(args)
    resolved = resolve_source(args)
    if resolved is None:
        return
    path, m, source_name, rest = resolved
    n, query, n_only = parse_chunk_args(rest)
    if query is not None and not query.strip():
        print("请先输入一个问题。")
        return
    if n <= 0:
        print(f"块大小必须大于 0，你输入的是：{n}")
        return
    conn = init_db()
    sid = resolve_sid(conn, mode, given_sid)
    if sid is not None:
        if conn.execute("SELECT 1 FROM session WHERE id = ?", (sid,)).fetchone() is None:
            print(f"会话 #{sid} 不存在。用 --sessions 查看已有会话，或不带 -s 新建会话。")
            return
        history = load_history(conn, sid)
        round_no = len(history) // 2 + 1
    else:
        history = []
        round_no = 1
    new_session = False
    if conn and query and sid is None:
        sid = start_session(conn, query)
        new_session = sid is not None
    if query and sid is not None:
        print(f"（会话 #{sid}，第 {round_no} 轮）")
    if m is None and not (BASE / "data").is_dir():
        fail_exit(conn, sid, new_session, f"资料目录不存在：{BASE / 'data'}")
        return
    t = load_chunks(m, n, source_name)
    print(f"文件:{path}")
    if m is not None:
        print(f"总字数:{len(m)}")
    print(f"切出块数:{len(t)}")
    if not t:
        where = "data 目录" if m is None else f"文件 {path}"
        fail_exit(conn, sid, new_session, f"资料为空：{where} 里没有可用的文本。")
        return
    need_model = bool(query) or (sys.stdin.isatty() and not n_only)
    bm25 = build_index(t) if need_model else None
    if need_model and bm25 is None:
        fail_exit(conn, sid, new_session,
                  "当前资料里没有可检索的内容（全是停用词或空白），请换一份资料或检查文件内容。")
        return
    if query:
        trace=[]
        t_start = time.time()
        answer, log = agent(query, t, bm25, trace, history=history)
        elapsed_all = time.time() - t_start
        if answer is None:
            if new_session:
                drop_empty_session(conn, sid)
            return
        render_round(answer, log, trace, elapsed_all)
        if conn and sid is not None:
            save_turn(conn, sid, query, answer)
    elif sys.stdin.isatty() and not n_only:
        interactive_loop(t, bm25, conn, sid, history)
    else:
        show_demo(t)

if __name__=="__main__":
    main()
