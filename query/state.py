"""
LangGraph state for PaperMind.

`PaperMindState` is the central state object passed between LangGraph nodes.
Fields like `retrieved_chunks` are annotated with `operator.add` so that results
from parallel branches can be merged by appending into a single list.
"""

from typing import TypedDict, List, Dict, Annotated
import operator


class PaperMindState(TypedDict):
    query: str
    sub_queries: List[str]
    retrieved_chunks: Annotated[List[dict], operator.add]
    failed_sub_queries: List[str]
    replan_count: int
    synthesis: str
    confidence_score: float