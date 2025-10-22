# serve.py
import os
from graph import build_graph
from db_utils import open_checkpointer
from ui import launch_ui

# 统一文件夹并用绝对路径
os.makedirs("./data", exist_ok=True)
DB_PATH = os.path.abspath("./data/graph.db")

checkpointer = open_checkpointer(DB_PATH)
print("[debug serve] checkpointer type:", type(checkpointer))
print("[debug serve] db_path:", DB_PATH)

APP = build_graph().compile(checkpointer=checkpointer)

if __name__ == "__main__":
    launch_ui(APP, DB_PATH, server_name="127.0.0.1", server_port=7860, inbrowser=True)
