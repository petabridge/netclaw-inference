"""Synthetic screenshots for the 0010 media prompt reuse tools: deterministic PNGs from a seed, standard library only.

Two seeds at the same size give the same visual grid (the same placeholder tokens) but different pixels, which is
exactly the case the prompt cache must tell apart.
"""
from __future__ import annotations

import base64
import random
import struct
import zlib


def png(width: int, height: int, seed: int) -> bytes:
    """An RGB PNG of window-like panels and text-like bars, different for every seed."""
    rng = random.Random(seed)
    bg = [rng.randrange(180, 256) for _ in range(3)]
    rows = [bytearray(bytes(bg) * width) for _ in range(height)]
    for _ in range(12):                                   # panels
        x0, y0 = rng.randrange(width), rng.randrange(height)
        x1, y1 = min(width, x0 + rng.randrange(40, max(41, width // 2))), min(height, y0 + rng.randrange(30, max(31, height // 2)))
        color = bytes(rng.randrange(0, 256) for _ in range(3))
        for y in range(y0, y1):
            rows[y][3 * x0:3 * x1] = color * (x1 - x0)
    for _ in range(60):                                   # text-like bars
        x0, y0 = rng.randrange(width), rng.randrange(height)
        x1, y1 = min(width, x0 + rng.randrange(20, 200)), min(height, y0 + rng.randrange(4, 10))
        color = bytes(rng.randrange(0, 90) for _ in range(3))
        for y in range(y0, y1):
            rows[y][3 * x0:3 * x1] = color * (x1 - x0)
    raw = b"".join(b"\x00" + bytes(r) for r in rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def part(width: int, height: int, seed: int) -> dict:
    """An OpenAI image_url content part with the PNG as a data URL (served with VISION_URLS=0)."""
    url = "data:image/png;base64," + base64.b64encode(png(width, height, seed)).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)
