# ui.py
import json
import uuid
import os
import sqlite3
import gradio as gr
from typing import Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# ---------- 历史会话列表（从 checkpoints 表读 thread_id） ----------
def list_threads(db_path: str):
    """读取 SQLite checkpointer 里的 thread_id 列表（按最近时间倒序）"""
    if not os.path.exists(db_path):
        return []

    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()

            # 1) 确认表存在
            t = cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
            ).fetchone()
            if not t:
                return []

            # 2) 找到可用的“时间列”
            cols = {row[1] for row in cur.execute("PRAGMA table_info(checkpoints)").fetchall()}
            time_col = None
            for c in ("created_at", "ts", "inserted_at", "updated_at"):
                if c in cols:
                    time_col = c
                    break

            # 3) 按时间排序拿每个 thread_id 的最新一条
            if time_col:
                rows = cur.execute(
                    f"""
                    SELECT thread_id, MAX({time_col}) AS last_ts
                    FROM checkpoints
                    GROUP BY thread_id
                    ORDER BY last_ts DESC
                    """
                ).fetchall()
            else:
                # 没有时间列就按 thread_id 倒序
                rows = cur.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC"
                ).fetchall()

            # rows 可能形如 [(tid, ts), ...] 或 [(tid,), ...]
            result = []
            for r in rows:
                if isinstance(r, (list, tuple)) and len(r) >= 1:
                    result.append(r[0])
            return result

    except Exception as e:
        print("[warn] list_threads failed:", e)
        return []

def new_thread_id() -> str:
    return f"conv_{uuid.uuid4().hex[:8]}"

# ---------- JSON 友好：把 BaseMessage 转 dict ----------
def _msg_to_dict(m: BaseMessage) -> dict:
    if isinstance(m, HumanMessage):
        return {"role": "user", "content": m.content}
    if isinstance(m, AIMessage):
        return {"role": "assistant", "content": m.content}
    # 其他类型也兜底
    return {"role": getattr(m, "type", "message"), "content": getattr(m, "content", "")}

def _state_to_jsonable(obj: Any):
    # 处理输出 state 中的 BaseMessage
    if isinstance(obj, BaseMessage):
        return _msg_to_dict(obj)
    if isinstance(obj, list):
        return [_state_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _state_to_jsonable(v) for k, v in obj.items()}
    return obj

# ---------- 单轮运行 ----------
def _run_once(app, thread_id: str, question: str, contract_text: str, force_branch: str):
    state_in = {"question": (question or "").strip()}
    if contract_text:
        state_in["contract_text"] = contract_text.strip()
    # 强制分支（仅调试）
    if force_branch in {"lookup", "review", "draft", "memory", "smalltalk"}:
        state_in["router"] = force_branch

    config = {"configurable": {"thread_id": (thread_id or "conv_default")}}
    out = app.invoke(state_in, config=config)

    # 友好展示
    debug = json.dumps(_state_to_jsonable(out), ensure_ascii=False, indent=2)
    return out.get("router", ""), out.get("response", ""), debug

# ---------- UI ----------
def create_ui(app, db_path: str):
    with gr.Blocks(title="法律助手 · 测试UI（LangGraph）") as demo:
        gr.Markdown("## 法律助手 · 测试UI（LangGraph）\n仅用于内部测试，输出为一般信息，非法律意见。")

        with gr.Row():
            # 左侧：历史会话栏
            with gr.Column(scale=3):
                gr.Markdown("### 历史会话")
                with gr.Row():
                    refresh_btn = gr.Button("刷新列表", variant="secondary")
                    new_btn = gr.Button("新建会话", variant="secondary")

                threads_list = gr.Radio(label="点击加载会话", choices=[], value=None, interactive=True)

            # 右侧：输入区
            with gr.Column(scale=7):
                with gr.Row():
                    question = gr.Textbox(
                        label="用户输入（问题/指令）", value="", lines=6,
                        placeholder="例如：起草一份NDA / 回顾一下我们刚才的操作"
                    )
                    force = gr.Dropdown(
                        choices=["", "lookup", "review", "draft", "memory", "smalltalk"],
                        value="", label="强制分支（调试用，仅显示）",
                    )

                contract = gr.Textbox(
                    label="合同文本/要素（可选）", lines=10,
                    placeholder="可以粘贴合同正文，或以“键:值”逐项填写"
                )

                thread_id = gr.Textbox(label="thread_id（会话标识）", value="conv_default")
                run_btn = gr.Button("运行", variant="primary")

        router_out = gr.Textbox(label="分类结果（router）", interactive=False)
        response_out = gr.Markdown(label="回答 / 草稿 / 审核结果")
        debug_out = gr.Code(label="调试状态（完整 state 输出）", language="json")

        # 事件
        def do_refresh():
            return gr.update(choices=list_threads(db_path), value=None)
        refresh_btn.click(fn=do_refresh, outputs=threads_list)

        threads_list.change(fn=lambda tid: tid or "conv_default", inputs=threads_list, outputs=thread_id)
        new_btn.click(fn=lambda: new_thread_id(), outputs=thread_id)

        run_btn.click(
            fn=lambda tid, q, ct, f: _run_once(app, tid, q, ct, f),
            inputs=[thread_id, question, contract, force],
            outputs=[router_out, response_out, debug_out],
        )

        demo.load(fn=do_refresh, outputs=threads_list)
    return demo

def launch_ui(app, db_path: str, server_name="127.0.0.1", server_port=7860, inbrowser=True):
    app_ui = create_ui(app, db_path)
    app_ui.launch(server_name=server_name, server_port=server_port, inbrowser=inbrowser)
