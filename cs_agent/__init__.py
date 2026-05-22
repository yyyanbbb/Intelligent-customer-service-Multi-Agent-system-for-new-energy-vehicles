from __future__ import annotations


def chat(*args, **kwargs):
    from cs_agent.graph import chat as _chat

    return _chat(*args, **kwargs)


def get_graph(*args, **kwargs):
    from cs_agent.graph import get_graph as _get_graph

    return _get_graph(*args, **kwargs)


def rag_retrieve(*args, **kwargs):
    from cs_agent.tools.rag_tool import rag_retrieve as _rag_retrieve

    return _rag_retrieve(*args, **kwargs)


def rebuild_index(*args, **kwargs):
    from cs_agent.tools.rag_tool import rebuild_index as _rebuild_index

    return _rebuild_index(*args, **kwargs)


__all__ = ["chat", "get_graph", "rag_retrieve", "rebuild_index"]
