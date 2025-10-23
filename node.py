# node.py
import os

from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage,SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
import requests

from state import AgentState
from typing import List
from langchain_core.messages import BaseMessage

def web_search(query: str, k: int = 5,*, debug: bool = True) -> str:
    """
    轻量 Web 检索；返回“资料要点 + 来源 URL”的纯文本。
    """
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        if debug:
            print("[lookup] Tavily API key 不存在，跳过联网搜索，改用模型兜底。")
        # 没 key 就直接不检索，交给 LLM 兜底
        return ""
    # 2) 直连 Tavily API 兜底
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "num_results": k,
                "search_depth": "basic",  # 或 "advanced"
            },
            timeout=15,
        )
        data = resp.json()
        items = data.get("results", []) if isinstance(data, dict) else []
        lines = [
            f"- {it.get('content', '')} (来源: {it.get('url', '')})"
            for it in items if it.get("content")
        ]
        return "\n".join(lines)
    except Exception:
        return ""

MAX_TURNS = 4  # 最近 4 轮

def _last_messages(state, turns: int = MAX_TURNS) -> List[BaseMessage]:
    msgs: List[BaseMessage] = state.get("messages", [])
    return msgs[-2 * turns:]     # 1 轮 = user+assistant 两条




# ============ 工具：把 messages 转成便于 LLM 消化/回显的文本 ============
def _messages_plaintext(messages: List[BaseMessage]) -> str:
    lines = []
    for m in messages or []:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)

# ============ 1) 意图分类（LLM + 规则兜底） ============
def intent_classification(state: AgentState, config: RunnableConfig, llm: BaseChatModel) -> Dict[str, Any]:
    q = (state.get("question") or "").strip()
    hist_txt = _messages_plaintext(state.get("messages", []))

    system = (
        "你是一个路由器。根据用户请求，将其分类为以下之一："
        "['memory','draft','review','lookup','smalltalk']。\n"
        "规则：\n"
        "- 包含“回顾/还记得/总结刚才/之前说过” → memory\n"
        "- 起草/生成合同 → draft\n"
        "- 审核/优化/修改合同 → review\n"
        "- 检索/查询/找法规/问法条 → lookup\n"
        "- 其他闲聊 → smalltalk\n"
        "仅输出一个JSON，如：{\"router\": \"draft\"}"
    )
    user = f"历史片段：\n{hist_txt}\n\n本轮问题：{q}\n请给出分类："

    try:
        resp = llm.invoke([("system", system), ("user", user)])
        text = (resp.content or "").strip()
    except Exception:
        text = ""

    import json, re
    tag = "unknown"
    if text:
        m = re.search(r'\"router\"\s*:\s*\"(memory|draft|review|lookup|smalltalk)\"', text)
        if m:
            tag = m.group(1)

    # 兜底的强规则
    lower = q.lower()
    if tag == "unknown":
        if any(k in q for k in ["回顾", "还记得", "回溯", "总结一下", "之前说"]):
            tag = "memory"
        elif any(k in q for k in ["起草", "生成", "草拟"]) and "合同" in q:
            tag = "draft"
        elif any(k in q for k in ["审核", "润色", "优化", "修改"]) and "合同" in q:
            tag = "review"
        elif any(k in q for k in ["查询", "检索", "找", "法规", "法条"]):
            tag = "lookup"
        else:
            tag = "smalltalk"

    # 把用户提问追加到对话
    out = {
        "router": tag,
        "messages": [HumanMessage(content=q)],
    }
    return out

# ============ 2) Memory 节点：回顾最近对话 ============
def memory_node(state: AgentState, *, llm: BaseChatModel) -> Dict[str, Any]:
    msgs: List[BaseMessage] = state.get("messages", [])
    # 只看最近 12 轮
    window = msgs[-24:]  # human/ai 共计
    prompt = (
        "请用简洁要点回顾最近几轮的对话，列出已完成的操作/决定（若没有就说“暂无明确动作”）：\n\n"
        f"{_messages_plaintext(window)}"
    )
    ai = llm.invoke([("user", prompt)])
    answer = (ai.content or "").strip() or "暂无可回顾的历史~"

    return {
        "response": answer,
        # 关键：把助手答复也 append 进 messages
        "messages": [AIMessage(content=answer)],
    }

# ============ 3) Draft / Review / Lookup / Smalltalk（示例实现） ============
def draft_node(state: AgentState, *, llm: BaseChatModel) -> Dict[str, Any]:
    q = state.get("question") or "请起草合同。"
    prompt = f"请根据要素起草一份简明的服务合同草案。输入：{q}"
    ai = llm.invoke([("user", prompt)])
    answer = (ai.content or "").strip()
    return {"response": answer, "messages": [AIMessage(content=answer)]}

def review_node(state: AgentState, *, llm: BaseChatModel) -> Dict[str, Any]:
    """
    合同审阅 / 风险提示节点：
    - 优先读取 contract_text；若为空，退回到 question（当作审阅对象或问题背景）
    - 输出结构化的风险意见，尽量引用条款号/关键原文（若有）
    """
    text = state.get("contract_text") or ""
    q = state.get("question") or ""

    # 若没有正文，至少保证模型有个审阅目标
    review_target = text.strip() or q.strip() or "（暂无合同行文，仅能给出一般性审阅建议）"

    system = SystemMessage(
        content=(
            "你是一名严谨的法律顾问，擅长合同审阅与风险识别。"
            "可以进行联网检索，除非用户提供的文本中就有证据，否则不要编造具体法条。"
            "若信息不足，请清楚写明假设、边界与需要用户补充的材料。"
        )
    )

    user = HumanMessage(
        content=(
            "请审阅以下合同文本（或与其相关的说明），给出专业且可执行的风险提示：\n\n"
            f"【审阅对象】\n{review_target}\n\n"
            "【输出要求】\n"
            "1. 以有序列表列出“风险点（Risk）”。每个风险点需包含：\n"
            "   - 严重程度：高/中/低（并简述理由）\n"
            "   - 依据/条款：若文本中有对应条款或关键句，引用原文或条款号（没有就写“文本未提供”）\n"
            "   - 建议与修改示例：给出具体修改建议；能给出示范性措辞更好\n"
            "   - 需补充材料：如因信息不足导致无法判断，列出需要的材料\n"
            "2. 若文本较短或不完整，请在开头用一句“信息完整性评估：…”说明信息是否充分。\n"
            "3. 最后给出“结论综述”：用3-5条总结最关键的注意事项与下一步建议。\n"
            "4. 不要编造引用；没有就写“文本未提供”。\n"
        )
    )

    ai = llm.invoke([system, user])
    answer = (ai.content or "").strip()

    return {
        "response": answer,
        "messages": [AIMessage(content=answer)],
    }
def lookup_node(state: AgentState, *, llm: BaseChatModel, web_search=None) -> Dict[str, Any]:
    """
    在线检索 + LLM 解释式回答。
    - 先用 web_search(query) 拉一段“资料要点+来源”纯文本
    - 再把资料、上下文、最近对话摘要一并给到 LLM，要求要点式输出与可操作建议
    """
    q = (state.get("question") or "").strip()

    # 最近对话摘要（可帮助 LLM 了解上下文）
    window = _last_messages(state, 4)
    history_hint = _messages_plaintext(window)

    # 可选的业务上下文（如你在别处塞了合同正文/拼好的上下文）
    context_text = (state.get("context_text") or state.get("contract_text") or "").strip()

    # 联网搜索（可关闭：把 web_search 传 None 即可）
    web_snippets = ""
    if web_search and q:
        try:
            web_snippets = web_search(q, k=5)
        except Exception:
            web_snippets = ""

    if (web_snippets != ""):
        print("找到了")

    system = SystemMessage(
        content=(
            "你是一名严谨的法律助手。可以参考联网检索到的资料，也可以依靠你的常识与法律知识回答。"
            "回答尽量结构化、要点式，并给出可操作建议；若不确定或资料不足，明确说明“不确定/需补充”，不要编造引用。"
        )
    )
    user = HumanMessage(
        content=(
            f"【用户问题】\n{q}\n\n"
            f"【最近对话摘要】\n{history_hint or '（无）'}\n\n"
            f"【可选业务上下文/合同行文】\n{context_text or '（无）'}\n\n"
            f"【检索到的资料（可能不完整，仅供参考）】\n{web_snippets or '（未检索到有效资料或暂时不可用）'}\n\n"
            "请综合以上信息，围绕问题给出要点式回答，并在最后给出下一步建议；对外部资料引用只需写出简短出处/URL。"
        )
    )

    ai = llm.invoke([system, user])
    answer = (ai.content or "").strip()

    return {
        "response": answer,
        "messages": [AIMessage(content=answer)],
    }

def smalltalk_node(state: AgentState, *, llm: BaseChatModel) -> Dict[str, Any]:
    q = state.get("question") or "你好"
    ai = llm.invoke([("user", q)])
    answer = (ai.content or "").strip()
    return {"response": answer, "messages": [AIMessage(content=answer)]}
