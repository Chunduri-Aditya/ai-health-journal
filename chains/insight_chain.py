"""
LangChain-based RAG insight chain.

This module wires the existing JSON-generation pipeline into a LangChain
Runnable graph, so we can describe the app as using LangChain for
RAG-style, grounded journaling analysis.
"""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

try:  # Optional LangChain Ollama integration
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    _OLLAMA_CHAIN_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    _OLLAMA_CHAIN_AVAILABLE = False


def _build_prompt(inputs: Dict[str, Any]) -> str:
    """Construct a grounded, empathy-focused prompt."""
    entry = inputs.get("entry", "")
    retrieved_context = inputs.get("retrieved_context", "") or "(no prior context retrieved)"

    return f"""
You are an emotionally intelligent journaling assistant.

User's current journal entry:
\"\"\"{entry}\"\"\"

Retrieved memory excerpts from the user's past entries:
\"\"\"{retrieved_context}\"\"\"

Your job:
- Respond with a concise, warm, grounded analysis.
- Be explicit about what is grounded in the entry or retrieved excerpts vs. what is a hypothesis.
- Avoid making claims that cannot be supported by the entry or retrieved context.

Return a JSON object with:
- summary: string
- emotions: string[]
- patterns: string[]
- triggers: string[]
- coping_suggestions: string[]
- quotes_from_user: string[]   # short direct quotes
- confidence: float            # 0.0–1.0
- grounding_evidence: string[] # excerpts you relied on
- uncertainties: string[]      # places where you are unsure or hypothesizing
"""


def _make_llm_caller(generator_model: str):
    """
    Construct a LangChain Runnable that calls Ollama via LangChain when available,
    otherwise fall back to the existing json_generate helper.
    """

    if _OLLAMA_CHAIN_AVAILABLE:
        # LangChain + Ollama path
        from schemas.analysis import AnalysisOutput

        parser = PydanticOutputParser(pydantic_object=AnalysisOutput)

        system_template = """
You are an emotionally intelligent journaling assistant.
You analyze the user's entry plus retrieved memory excerpts, and you must:
- Be empathetic and grounded.
- Only make claims supported by the entry or retrieved excerpts.
- Call out any hypotheses explicitly in 'uncertainties'.

Return ONLY a JSON object matching the provided schema.
{format_instructions}
"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                (
                    "human",
                    "User entry:\n{entry}\n\nRetrieved context:\n{retrieved_context}\n",
                ),
            ]
        )

        llm = ChatOllama(model=generator_model, temperature=0.0)

        chain = prompt | llm | parser

        def _call_llm(inputs: Dict[str, Any]) -> Dict[str, Any]:
            return chain.invoke(
                {
                    "entry": inputs.get("entry", ""),
                    "retrieved_context": inputs.get("retrieved_context", ""),
                    "format_instructions": parser.get_format_instructions(),
                }
            ).model_dump()

        return _call_llm

    # Fallback to json_generate helper if LangChain Ollama isn't available
    from llm_client import json_generate, DRAFT_JSON_SCHEMA, DRAFT_SYSTEM_PROMPT

    def _call_llm(inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt: str = inputs["prompt"]
        return json_generate(
            generator_model,
            DRAFT_SYSTEM_PROMPT,
            prompt,
            max_retries=5,
            json_schema=DRAFT_JSON_SCHEMA,
        )

    return _call_llm


def build_insight_chain(generator_model: str):
    """
    Build a LangChain Runnable that:
    - accepts {entry, retrieved_context}
    - formats a grounded, empathy-focused prompt
    - calls the existing JSON generation helper
    """
    llm_caller = _make_llm_caller(generator_model)

    if _OLLAMA_CHAIN_AVAILABLE:
        # LangChain-native chain already handles prompt construction
        return RunnablePassthrough() | RunnableLambda(llm_caller)

    # Fallback path: manually build prompt then call json_generate
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda inputs: {**inputs, "prompt": _build_prompt(inputs)})
        | RunnableLambda(llm_caller)
    )


def run_insight_chain(entry: str, retrieved_context: str, generator_model: str) -> Dict[str, Any]:
    """
    Convenience helper: run the chain end-to-end.

    Not currently wired into the main Flask route by default, but can be
    used from eval tooling or alternate endpoints.
    """
    chain = build_insight_chain(generator_model)
    return chain.invoke({"entry": entry, "retrieved_context": retrieved_context})

