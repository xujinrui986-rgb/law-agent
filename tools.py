import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
from typing import Optional, List
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # 确保读取 .env

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "qwen3-max"),
    temperature=float(os.getenv("TEMPERATURE", "0")),
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),          # 用 DashScope 的 key
    base_url=os.getenv("BASE_URL"),               # https://dashscope.aliyuncs.com/compatible-mode/v1
)







