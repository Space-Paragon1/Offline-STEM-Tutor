from __future__ import annotations

def complexity_hint(topic: str) -> str:
    t = (topic or "").lower()
    if "binary search" in t:
        return "Binary search runs in O(log n) time on a sorted array."
    if "merge sort" in t:
        return "Merge sort runs in O(n log n) time and uses O(n) extra space."
    if "hash" in t:
        return "Hash table average lookup is O(1), worst-case O(n)."
    return "Share the algorithm or code and I’ll estimate time/space complexity."
