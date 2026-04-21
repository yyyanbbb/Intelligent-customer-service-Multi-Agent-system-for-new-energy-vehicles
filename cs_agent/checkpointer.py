"""
AsyncSqliteSaver 封装。把 async loop 放到后台 daemon 线程里跑，
这样 graph.invoke()（同步）和 graph.astream_events()（Gradio 的 async loop）
可以共享同一个 checkpointer 实例，不会互相抢 event loop。
"""
from __future__ import annotations
import asyncio
import threading
from pathlib import Path

_DB_PATH = Path(__file__).parent / "knowledge" / "checkpoints.db"
_checkpointer_instance = None
_bg_loop: asyncio.AbstractEventLoop | None = None
_init_done = threading.Event()
_init_error: Exception | None = None


def _start_background_loop():
    """后台 daemon 线程入口，独占一个 SelectorEventLoop。"""
    global _checkpointer_instance, _bg_loop, _init_error

    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    _bg_loop = loop

    async def _init():
        global _checkpointer_instance, _init_error
        try:
            _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(str(_DB_PATH))
            saver = AsyncSqliteSaver(conn)
            await saver.setup()
            _checkpointer_instance = saver
        except Exception as e:
            _init_error = e
        finally:
            _init_done.set()

    loop.run_until_complete(_init())
    loop.run_forever()  # Keep the loop alive for run_coroutine_threadsafe calls


_bg_thread: threading.Thread | None = None
_start_lock = threading.Lock()


def _ensure_started():
    global _bg_thread
    with _start_lock:
        if _bg_thread is None:
            _bg_thread = threading.Thread(target=_start_background_loop, daemon=True)
            _bg_thread.start()


def get_checkpointer():
    """返回 AsyncSqliteSaver 单例，首次调用会阻塞直到初始化完成（最多 30s）。"""
    _ensure_started()
    _init_done.wait(timeout=30)
    if _init_error:
        raise RuntimeError(f"AsyncSqliteSaver init failed: {_init_error}") from _init_error
    if _checkpointer_instance is None:
        raise TimeoutError("AsyncSqliteSaver init timed out after 30s")
    return _checkpointer_instance
