# GateGuard: callers service/main.py, tests/test_agent_routing.py. User: Implement the plan as specified. Do NOT edit the plan file.
"""LangGraph state machine for the journal agent.

Nodes:
  - route: decide retrieve / metadata / propose_write / confirm / respond
  - retrieve / metadata / propose_write / apply_or_cancel / respond

Conditional routing is real (not a single tool-in-a-loop). Write actions
always pass through the confirmation gate before mutation.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Literal, Optional

from .config import load_config
from .providers.factory import get_llm_provider
from .vector_store.base import VectorStore
from .vector_store.factory import get_vector_store

from .agent.confirmation import is_confirmation, is_denial
from .agent.intent import classify_intent
from .agent.state import AgentState
from .agent.respond import generate_grounded_response
from .agent.tools import (
    tool_apply_write,
    tool_propose_write,
    tool_query_metadata,
    tool_retrieve,
)

logger = logging.getLogger(__name__)

_GRAPH_CACHE: Dict[tuple[int, str], Any] = {}


def _resolve_namespace(state: AgentState, fallback: str) -> str:
    return state.get("namespace") or fallback


_META = re.compile(r"\b(metadata|list entries|how many|entry id|when did)\b", re.I)
_RETRIEVE = re.compile(
    r"\b(find|search|recall|retrieve|remember|what did i|look up)\b", re.I
)
_WRITE = re.compile(
    r"\b(write|save|store|add|append|log this|remember this)\b", re.I
)

def _strip_write_directive(text: str) -> str:
    """Remove leading save/write directives; keep the journal payload clean."""
    cleaned = re.sub(
        r"^\s*(please\s+)?(write|save|store|add|append|log|remember)(\s+this)?\s*:?\s*",
        "",
        text or "",
        flags=re.I,
    ).strip()
    return cleaned or (text or "").strip()


def _apply_result_message(result: Dict[str, Any]) -> str:
    if result.get("status") == "applied":
        return f"Write applied to {result.get('entry_id')}."
    if result.get("reason") == "retrieval_disabled":
        return "Write failed: retrieval is disabled, so entries cannot be stored."
    return "Write failed."




def _last_user_text(state: AgentState) -> str:
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "type", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if role in ("user", "human") or (isinstance(m, dict) and m.get("role") == "user"):
            return content or ""
        # LangChain HumanMessage.type == "human"
        if getattr(m, "type", None) == "human":
            return content or ""
    if msgs:
        m = msgs[-1]
        return m.get("content") if isinstance(m, dict) else getattr(m, "content", "") or ""
    return ""


def _append_trace(state: AgentState, event: Dict[str, Any]) -> list:
    trace = list(state.get("tool_trace") or [])
    trace.append(event)
    return trace


def route_intent(state: AgentState) -> Literal[
    "confirm_gate", "retrieve", "metadata", "propose_write", "respond"
]:
    if state.get("awaiting_confirmation") and state.get("pending_write"):
        return "confirm_gate"
    text = _last_user_text(state)
    if _META.search(text):
        return "metadata"
    if _RETRIEVE.search(text):
        return "retrieve"
    if _WRITE.search(text):
        return "propose_write"
    return "respond"


def build_journal_agent(
    store: Optional[VectorStore] = None,
    *,
    namespace: Optional[str] = None,
):
    """Compile the LangGraph StateGraph. Lazy-imports langgraph."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as e:
        raise RuntimeError(
            "langgraph is required for the journal agent. "
            "Install: pip install -r requirements-service.txt"
        ) from e

    cfg = load_config()
    vs = store or get_vector_store()
    ns_default = namespace or cfg.rag_namespace_fixed
    provider = get_llm_provider(cfg)

    def node_retrieve(state: AgentState) -> Dict[str, Any]:
        text = _last_user_text(state)
        ns = _resolve_namespace(state, ns_default)
        result = tool_retrieve(vs, text, namespace=ns, top_k=cfg.retrieval_top_k)
        context = result.get("context") or ""
        if not context:
            return {
                "last_tool": "retrieve_journal",
                "tool_trace": _append_trace(state, result),
                "final_response": "No matching journal entries found.",
            }
        # Synthesize hits into a grounded summary
        quality = generate_grounded_response(
            provider,
            cfg,
            user_message=text,
            retrieved_context=context,
        )
        return {
            "last_tool": "retrieve_journal",
            "tool_trace": _append_trace(state, result),
            "final_response": quality.get("answer") or context,
        }

    def node_metadata(state: AgentState) -> Dict[str, Any]:
        ns = _resolve_namespace(state, ns_default)
        result = tool_query_metadata(vs, namespace=ns)
        preview = "\n".join(
            f"- {r['entry_id']}: {r.get('preview') or ''}" for r in result["rows"][:10]
        ) or "(no entries)"
        return {
            "last_tool": "query_entry_metadata",
            "tool_trace": _append_trace(state, result),
            "final_response": f"Entry metadata ({len(result['rows'])}):\n{preview}",
        }

    def node_propose_write(state: AgentState) -> Dict[str, Any]:
        text = _last_user_text(state)
        cleaned = _strip_write_directive(text)
        ns = _resolve_namespace(state, ns_default)
        result = tool_propose_write(text=cleaned, namespace=ns)
        if not cfg.agent_confirm_writes:
            applied = tool_apply_write(vs, result["pending_write"])
            return {
                "last_tool": "apply_write",
                "pending_write": None,
                "awaiting_confirmation": False,
                "tool_trace": _append_trace(state, {**result, **applied}),
                "final_response": _apply_result_message(applied),
            }
        return {
            "last_tool": "propose_write",
            "pending_write": result["pending_write"],
            "awaiting_confirmation": True,
            "tool_trace": _append_trace(state, result),
            "final_response": result["message"],
        }

    def node_confirm_gate(state: AgentState) -> Dict[str, Any]:
        text = _last_user_text(state)
        pending = state.get("pending_write")
        if not pending:
            return {
                "awaiting_confirmation": False,
                "final_response": "Nothing pending to confirm.",
            }
        if is_confirmation(text):
            result = tool_apply_write(vs, pending)
            return {
                "pending_write": None,
                "awaiting_confirmation": False,
                "last_tool": "apply_write",
                "tool_trace": _append_trace(state, result),
                "final_response": _apply_result_message(result),
            }
        if is_denial(text):
            event = {"tool": "apply_write", "status": "cancelled"}
            return {
                "pending_write": None,
                "awaiting_confirmation": False,
                "last_tool": "apply_write",
                "tool_trace": _append_trace(state, event),
                "final_response": "Write cancelled.",
            }
        # Neither confirm nor deny: supersede pending write and continue normal routing
        event = {"tool": "apply_write", "status": "cancelled", "reason": "superseded"}
        # Clear pending and route to normal intent
        state["pending_write"] = None
        state["awaiting_confirmation"] = False
        state["tool_trace"] = _append_trace(state, event)
        # Route to normal intent handling
        intent = route_intent(state)
        if intent == "retrieve":
            return node_retrieve(state)
        elif intent == "metadata":
            return node_metadata(state)
        elif intent == "propose_write":
            return node_propose_write(state)
        else:
            return node_respond(state)

    def node_respond(state: AgentState) -> Dict[str, Any]:
        text = _last_user_text(state)
        ns = _resolve_namespace(state, ns_default)
        retrieval = tool_retrieve(vs, text, namespace=ns, top_k=cfg.retrieval_top_k)
        context = retrieval.get("context") or ""
        quality = generate_grounded_response(
            provider,
            cfg,
            user_message=text,
            retrieved_context=context,
        )
        event = {
            **retrieval,
            "quality": {
                "verified": quality.get("verified"),
                "revised": quality.get("revised"),
                "model": quality.get("model"),
                "verdict": quality.get("verdict"),
            },
        }
        return {
            "last_tool": "respond",
            "tool_trace": _append_trace(state, event),
            "final_response": quality.get("answer") or "",
        }

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("metadata", node_metadata)
    graph.add_node("propose_write", node_propose_write)
    graph.add_node("confirm_gate", node_confirm_gate)
    graph.add_node("respond", node_respond)

    # Router: (a) if awaiting confirm/deny -> confirm_gate; (b) try LLM classify_intent; (c) fallback to regex route_intent
    def smart_route(state: AgentState) -> Literal["confirm_gate", "retrieve", "metadata", "propose_write", "respond"]:
        if state.get("awaiting_confirmation") and state.get("pending_write"):
            return "confirm_gate"
        text = _last_user_text(state)
        try:
            intent = classify_intent(provider, cfg, text)
            return "propose_write" if intent == "write" else intent  # type: ignore[return-value]
        except Exception:
            return route_intent(state)

    graph.set_conditional_entry_point(
        smart_route,
        {
            "retrieve": "retrieve",
            "metadata": "metadata",
            "propose_write": "propose_write",
            "confirm_gate": "confirm_gate",
            "respond": "respond",
        },
    )
    for node in ("retrieve", "metadata", "propose_write", "confirm_gate", "respond"):
        graph.add_edge(node, END)

    return graph.compile()


def run_agent_turn(
    user_text: str,
    *,
    store: Optional[VectorStore] = None,
    namespace: Optional[str] = None,
    prior_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute one user turn against the compiled graph."""
    vs = store or get_vector_store()
    ns = namespace or load_config().rag_namespace_fixed
    cache_key = (id(vs), ns)
    if cache_key not in _GRAPH_CACHE:
        _GRAPH_CACHE[cache_key] = build_journal_agent(store=vs, namespace=ns)
    app = _GRAPH_CACHE[cache_key]
    state: Dict[str, Any] = dict(prior_state or {})
    # Supersede pending write when the user changes topic (neither confirm nor deny).
    if state.get("awaiting_confirmation") and state.get("pending_write"):
        if not is_confirmation(user_text) and not is_denial(user_text):
            trace = list(state.get("tool_trace") or [])
            trace.append({"tool": "apply_write", "status": "cancelled", "reason": "superseded"})
            state["tool_trace"] = trace
            state["pending_write"] = None
            state["awaiting_confirmation"] = False
    state["messages"] = list(state.get("messages") or []) + [
        {"role": "user", "content": user_text}
    ]
    if "namespace" not in state:
        state["namespace"] = namespace or load_config().rag_namespace_fixed
    out = app.invoke(state)
    return out
