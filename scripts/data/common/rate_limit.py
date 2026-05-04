from __future__ import annotations

import threading
import time


class TokenBucket:
    """Einfacher Token-Bucket-RateLimiter.

    capacity: Max Tokens; refill_rate: Tokens pro Sekunde.
    """

    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, amount: float = 1.0):
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._tokens = min(
                    self.capacity, self._tokens + elapsed * self.refill_rate
                )
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
            # warten, bis 1 Token verfügbar ist
            time.sleep(max(0.05, amount / self.refill_rate))
