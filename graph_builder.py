from langgraph.graph import StateGraph, START, END
from nodes import load_and_split, extract_anchor, call_llm, parse_and_repair, finalize, save_full_state, validate_items

def build_graph():
    g = StateGraph(dict)
    g.add_node("load", load_and_split)
    g.add_node("anchor", extract_anchor)
    g.add_node("llm", call_llm)
    g.add_node("parse", parse_and_repair)
    g.add_node("final", finalize)
    g.add_node("save", save_full_state)
    g.add_node("validate", validate_items)

    g.add_edge(START, "load")
    g.add_edge("load", "anchor")
    g.add_edge("anchor", "llm")
    g.add_edge("llm", "parse")
    g.add_conditional_edges(
        "parse",
        lambda s: "llm" if s["current_idx"] < len(s["chunks"]) else "validate",
        {"llm": "llm", "validate": "validate"}
    )
    g.add_edge("validate", "final")
    g.add_edge("final", "save")
    g.add_edge("save", END)

    return g.compile()
