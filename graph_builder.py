from langgraph.graph import StateGraph, START, END
from nodes import load_and_split, extract_anchor, call_llm, parse_and_repair, finalize, save_full_state, validate_items, save_skipped_component, filter_chunks

def build_graph():
    g = StateGraph(dict)
    g.add_node("load", load_and_split)
    g.add_node("anchor", extract_anchor)
    g.add_node("filter", filter_chunks)
    g.add_node("llm", call_llm)
    g.add_node("parse", parse_and_repair)
    g.add_node("final", finalize)
    g.add_node("save", save_full_state)
    g.add_node("validate", validate_items)
    g.add_node("save_skipped", save_skipped_component)

    g.add_edge(START, "load")
    g.add_edge("load", "anchor")
    g.add_conditional_edges(
        "anchor",
        lambda s: "filter" if s.get("chunks") else "save_skipped",
        {"filter": "filter", "save_skipped": "save_skipped"}
    )
    g.add_edge("save_skipped", END)
    g.add_edge("filter", "llm")
    g.add_edge("llm", "parse")
    g.add_conditional_edges(
        "parse",
        lambda s: "llm" if s["current_idx"] < len(s["final_chunks"]) else "validate",
        {"llm": "llm", "validate": "validate"}
    )
    g.add_edge("validate", "final")
    g.add_edge("final", "save")
    g.add_edge("save", END)

    return g.compile()
