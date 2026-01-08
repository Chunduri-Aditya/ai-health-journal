#!/usr/bin/env python3
"""
Benchmark script for AI Health Journal.
Measures latency of /analyze endpoint with local Ollama.

Usage:
    python scripts/benchmark.py

Prerequisites:
    - Ollama must be running (ollama serve)
    - phi3:3.8b model must be available
"""

import requests
import time
import statistics
from typing import List

BASE_URL = "http://127.0.0.1:5000"
SAMPLE_ENTRY = "I've been feeling overwhelmed with work lately. There's so much to do and I don't know where to start. I feel like I'm constantly behind and can't catch up."
WARMUP_ITERATIONS = 2
MEASURE_ITERATIONS = 10


def check_ollama() -> bool:
    """Check if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def check_app() -> bool:
    """Check if Flask app is running."""
    try:
        response = requests.get(f"{BASE_URL}/ping", timeout=2)
        return response.status_code == 200 and response.json().get("status") == "ok"
    except requests.exceptions.RequestException:
        return False


def measure_latency() -> float:
    """Measure single request latency."""
    start = time.time()
    try:
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"entry": SAMPLE_ENTRY},
            timeout=30
        )
        elapsed = time.time() - start
        if response.status_code == 200:
            return elapsed
        else:
            print(f"  ⚠️  Request failed: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        print(f"  ⚠️  Request timed out")
        return None
    except Exception as e:
        print(f"  ⚠️  Error: {e}")
        return None


def main():
    print("🧠 AI Health Journal - Benchmark Tool\n")
    
    # Check prerequisites
    print("Checking prerequisites...")
    if not check_ollama():
        print("❌ Ollama is not running. Please start it with: ollama serve")
        return
    print("✅ Ollama is running")
    
    if not check_app():
        print("❌ Flask app is not running. Please start it with: python app.py")
        return
    print("✅ Flask app is running\n")
    
    # Warmup
    print(f"Warming up ({WARMUP_ITERATIONS} requests)...")
    for i in range(WARMUP_ITERATIONS):
        print(f"  Warmup {i+1}/{WARMUP_ITERATIONS}...", end="\r")
        measure_latency()
    print("  Warmup complete\n")
    
    # Measure
    print(f"Running {MEASURE_ITERATIONS} measurement requests...")
    latencies: List[float] = []
    for i in range(MEASURE_ITERATIONS):
        print(f"  Request {i+1}/{MEASURE_ITERATIONS}...", end="\r")
        latency = measure_latency()
        if latency is not None:
            latencies.append(latency)
        time.sleep(0.5)  # Small delay between requests
    
    print(f"  Completed {len(latencies)}/{MEASURE_ITERATIONS} successful requests\n")
    
    if not latencies:
        print("❌ No successful requests. Check Ollama and app status.")
        return
    
    # Calculate statistics
    mean_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    sorted_latencies = sorted(latencies)
    p95_index = int(len(sorted_latencies) * 0.95)
    p95_latency = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1]
    
    # Print results
    print("📊 Results:")
    print(f"  Mean latency:   {mean_latency:.2f}s")
    print(f"  Median latency: {median_latency:.2f}s")
    print(f"  P95 latency:    {p95_latency:.2f}s")
    print(f"  Min latency:     {min(latencies):.2f}s")
    print(f"  Max latency:     {max(latencies):.2f}s")
    print(f"\n  Sample entry length: {len(SAMPLE_ENTRY)} characters")
    print(f"  Model: phi3:3.8b (default)")


if __name__ == "__main__":
    main()
