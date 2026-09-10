from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """LangGraph state for the journal agent.

    `messages` accumulates chat turns. `pending_write` holds a write that
    requires explicit confirmation before it is applied. `tool_trace` is
    appended for observability (Langfuse / SSE).

    Note: we keep this as a plain TypedDict (no langgraph Annotated reducers)
    so routing/unit tests import cleanly without langgraph installed. The
    compiled graph treats messages as an overwrite list per turn via
    run_agent_turn merging.
    """

    messages: list
    namespace: str
    pending_write: Optional[Dict[str, Any]]
    awaiting_confirmation: bool
    last_tool: Optional[str]
    tool_trace: List[Dict[str, Any]]
    final_response: Optional[str]
