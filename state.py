# state.py
from typing import Dict, Any, List, Optional, Literal
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

# ======= 证据/引用（可保留，后续做 RAG 时用） =======
class Evidence(BaseModel):
    doc_id: str
    title: Optional[str] = None
    snippet: str
    source: Optional[str] = None
    url: Optional[str] = None
    score: Optional[float] = None

class Citation(BaseModel):
    doc_id: str
    locator: Optional[str] = None
    text: Optional[str] = None

Intent = Literal["lookup", "review", "draft", "smalltalk", "memory", "unknown"]

# ======= 关键：继承 MessagesState（自带 messages 列表 + add_messages 语义） =======
class AgentState(MessagesState):
    question: str
    router: Optional[Intent] = None
    context_text: Optional[str] = None
    response: Optional[str] = None
    evidences: List[Evidence] = []
    citations: List[Citation] = []
    risk: Optional[Literal["low","medium","high"]] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
