#!/usr/bin/env python3
"""
Pinecone bootstrap utility for AI Health Journal.

Creates or validates the configured Pinecone index so you can enable
VECTOR_BACKEND=pinecone with one command.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from pinecone import Pinecone, ServerlessSpec


def _env(name: str, default: Optional[str] = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        print(f"❌ Missing required env: {name}")
        sys.exit(1)
    return val


def main() -> None:
    api_key = _env("PINECONE_API_KEY")
    index_name = _env("PINECONE_INDEX")
    dim_str = _env("PINECONE_DIM", "384")  # all-MiniLM-L6-v2 default
    metric = _env("PINECONE_METRIC", "cosine")

    try:
        dimension = int(dim_str)
    except ValueError:
        print(f"❌ PINECONE_DIM must be an integer, got: {dim_str!r}")
        sys.exit(1)

    pc = Pinecone(api_key=api_key)

    existing = {info["name"]: info for info in pc.list_indexes()}

    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}' (dim={dimension}, metric={metric})...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("✅ Index created.")
    else:
        info = existing[index_name]
        config = info.get("dimension"), info.get("metric")
        if config != (dimension, metric):
            print(
                f"❌ Existing index '{index_name}' has dimension={info.get('dimension')}, "
                f"metric={info.get('metric')} but env expects dimension={dimension}, metric={metric}."
            )
            sys.exit(1)
        print(f"✅ Index '{index_name}' exists and matches env configuration.")

    desc = pc.describe_index(index_name)
    host = desc.host

    namespace = os.getenv("PINECONE_NAMESPACE", "ai-health-journal")

    print("\nNext steps:")
    print("1) Add or update these .env settings:")
    print(f"   PINECONE_API_KEY=***")
    print(f"   PINECONE_INDEX={index_name}")
    print(f"   PINECONE_NAMESPACE={namespace}")
    print(f"   PINECONE_DIM={dimension}")
    print(f"   PINECONE_METRIC={metric}")
    print("   VECTOR_BACKEND=pinecone")
    print("\n2) Optional privacy flags:")
    print("   PINECONE_STORE_TEXT=false  # embeddings + minimal metadata only (recommended)")
    print("\n3) Start the app and confirm /models shows vector_backend='pinecone'.")
    print(f"\nIndex host: {host}")


if __name__ == "__main__":
    main()

