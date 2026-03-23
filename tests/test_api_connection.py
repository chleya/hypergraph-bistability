#!/usr/bin/env python
"""Quick test script to verify MiniMax API connection."""

import os
import json
import httpx

# Set API key
API_KEY = 'sk-cp-32X4yB3hv4uMfzdBmke7EyaxE2pXmHkAGisoBxm1bTlSnUKXcH3lGRgWYcD62Nre5AacJpbi0E5yOx92m5rkIth9HioW2aCHP5r3LeCKBuf-wdr1TVgeFxY'
BASE_URL = 'https://api.minimaxi.com/anthropic'

print("Testing MiniMax API connection...")
print("-" * 50)

# Test 1: Check API key format
print(f"1. API Key length: {len(API_KEY)}")
print(f"2. Base URL: {BASE_URL}")

# Test 2: Try to get models
try:
    response = httpx.get(
        f"{BASE_URL}/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=15
    )
    print(f"3. /v1/models Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Available models: {json.dumps(response.json(), indent=2)[:300]}")
except Exception as e:
    print(f"3. /v1/models Error: {type(e).__name__}: {e}")

# Test 3: Try a simple message (if key works)
try:
    payload = {
        "model": "MiniMax-M2.7",
        "messages": [{"role": "user", "content": "Hi, how are you?"}],
        "max_tokens": 100
    }
    response = httpx.post(
        f"{BASE_URL}/v1/messages",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=30
    )
    print(f"4. /v1/messages Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        # Extract text from content
        texts = [c.get('text', '') for c in result.get('content', []) if c.get('type') == 'text']
        thinking = [c.get('thinking', '')[:50] for c in result.get('content', []) if c.get('type') == 'thinking']
        print(f"   Thinking: {thinking}")
        print(f"   Text: {texts}")
        print(f"   Model: {result.get('model')}")
        print(f"   Usage: {result.get('usage')}")
    else:
        print(f"   Error: {response.text[:200]}")
except Exception as e:
    print(f"4. /v1/messages Error: {type(e).__name__}: {e}")

print("-" * 50)
print("Done.")
