# db_utils.py
import os
from langgraph.checkpoint.sqlite import SqliteSaver

# db_utils.py
import os
import atexit
from langgraph.checkpoint.sqlite import SqliteSaver

def open_checkpointer(db_path: str):
    db_path = os.path.abspath(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    cm_or_inst = SqliteSaver.from_conn_string(db_path)
    if hasattr(cm_or_inst, "__enter__") and not hasattr(cm_or_inst, "get_next_version"):
        saver = cm_or_inst.__enter__()
        # 进程退出时关闭
        atexit.register(cm_or_inst.__exit__, None, None, None)
        return saver





