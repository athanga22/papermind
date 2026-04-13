"""
PaperMind LangGraph agent wiring.

Graph:
  planner -> dispatch_retrieval -> (retrieve_one x N in parallel) -> gate
  gate -> {replan | synthesize}
  replan -> dispatch_retrieval
  synthesize -> END
"""

from __future__ import annotations

from typing import Any, List

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from query.nodes.gate import gate_node, gate_route
from query.nodes.planner import planner_node
from query.nodes.replan import replan_node
from query.nodes.retrieval import retrieve_one_node
from query.nodes.synthesis import synthesis_node
from query.state import PaperMindState


def dispatch_retrieval(state: PaperMindState) -> List[Send]:
    """
    Fan-out dispatcher.

    Creates one parallel `retrieve_one` task per sub-query by sending each branch
    a state where `sub_queries=[that_sub_query]`.
    """
    sub_queries = list(state.get("sub_queries") or [])
    sends: list[Send] = []
    for sq in sub_queries:
        sends.append(Send("retrieve_one", {"sub_queries": [sq]}))
    return sends


def build_app():
    workflow = StateGraph(PaperMindState)

    # Nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("dispatch_retrieval", dispatch_retrieval)
    workflow.add_node("retrieve_one", retrieve_one_node)
    workflow.add_node("gate", gate_node)
    workflow.add_node("replan", replan_node)
    workflow.add_node("synthesize", synthesis_node)

    # Edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "dispatch_retrieval")

    # Fan-out: dispatch -> retrieve_one (parallel) -> gate
    workflow.add_conditional_edges("dispatch_retrieval", dispatch_retrieval, ["retrieve_one"])
    workflow.add_edge("retrieve_one", "gate")

    # Gate routing
    workflow.add_conditional_edges(
        "gate",
        gate_route,
        {
            "replan": "replan",
            "synthesize": "synthesize",
        },
    )
    workflow.add_edge("replan", "dispatch_retrieval")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


app = build_app()

