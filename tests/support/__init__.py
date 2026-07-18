"""Shared, deterministic test/eval doubles for the AI Health Journal.

These let the conversation and eval suites drive the real `/analyze` route and
the RAG memory loop without a live Ollama (fake provider) and, on the fast
path, without an embedding model (in-memory fake vector store). Real-Chroma
semantic checks live behind the `slow` marker and in the scenario runner.
"""
