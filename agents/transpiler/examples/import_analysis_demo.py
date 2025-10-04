"""
Import Analysis Demo
Shows how the import analyzer detects Python imports and maps them to Rust
"""

# Standard library imports - Full WASM compatible
import json
import logging
import re
from decimal import Decimal

# File I/O - Requires WASI
from pathlib import Path
import tempfile

# Date/Time - Requires JS interop
from datetime import datetime, timedelta
import time

# Networking - Requires JS interop
import http.client
from urllib.request import urlopen

# Async - Requires JS interop
import asyncio

# Cryptography - Full WASM (hashlib) + JS interop (secrets)
import hashlib
import secrets

# Data structures - Full WASM
from collections import deque
import itertools

# CLI - Full WASM
import argparse

# Testing - Full WASM
import unittest

def example_function():
    """Example function using various imports"""
    # JSON serialization
    data = {"name": "Alice", "score": 100}
    json_str = json.dumps(data)

    # Logging
    logging.info("Processing data")

    # Decimal math
    price = Decimal("19.99")

    # Path operations
    p = Path("data.txt")

    # Date/time
    now = datetime.now()
    delta = timedelta(days=1)

    # Regex
    pattern = re.compile(r"\d+")

    # Hash
    h = hashlib.sha256(b"data")

    # Collections
    q = deque([1, 2, 3])

    return json_str
