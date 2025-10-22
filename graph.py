# graph.py
import os
from functools import partial
from typing import Literal
from langgraph.graph import StateGraph, END
from tools import llm

from state import AgentState
from node import (
    intent_classification,
    memory_node,
    draft_node,
    review_node,
    lookup_node,
    smalltalk_node,
    web_search
)

# 你自己的模型（按你现有环境改）



def build_graph():
    g = StateGraph(AgentState)


    # ---- 节点注册 ----
    g.add_node("classify", partial(intent_classification, llm=llm))
    g.add_node("memory", partial(memory_node, llm=llm))
    g.add_node("draft", partial(draft_node, llm=llm))
    g.add_node("review", partial(review_node, llm=llm))
    g.add_node("lookup",   partial(lookup_node, llm=llm, web_search=web_search))  # ← 传入
    g.add_node("smalltalk", partial(smalltalk_node, llm=llm))

    # 入口：先分类
    g.set_entry_point("classify")

    # 根据 router 走向
    def route_by_intent(state: AgentState) -> Literal["memory","draft","review","lookup","smalltalk"]:
        tag = (state.get("router") or "smalltalk").lower()
        if tag not in {"memory","draft","review","lookup","smalltalk"}:
            tag = "smalltalk"
        return tag

    g.add_conditional_edges("classify", route_by_intent, {
        "memory": "memory",
        "draft": "draft",
        "review": "review",
        "lookup": "lookup",
        "smalltalk": "smalltalk",
    })

    # 每条边都到 END（单轮）
    for nd in ["memory","draft","review","lookup","smalltalk"]:
        g.add_edge(nd, END)

    return g
