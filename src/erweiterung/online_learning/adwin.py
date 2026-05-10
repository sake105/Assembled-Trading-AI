"""ADaptive WINdowing (ADWIN) für Concept-Drift-Detection.

Reference
---------
Bifet, A. & Gavaldà, R. (2007). Learning from time-changing data with
adaptive windowing. SDM 2007.

Idee
----
Halte ein Window W der Recent-Beobachtungen. Suche nach einem Split-Punkt
i, so dass die Mittelwerte vor und nach i statistisch verschieden sind
(Hoeffding-Bound). Wenn ja: drop die ältere Hälfte → Drift detected.

ADWIN garantiert:
- Zuverlässige Drift-Detection mit erwarteter Detection-Time O(σ²/Δ²).
- Adaptives Fenster — keine fixe Lookback-Wahl.

Anwendung
---------
- Trigger Re-Training von ML-Modellen wenn Concept-Drift erkannt.
- Regime-Change-Indikator.
"""

from __future__ import annotations

import math
from collections import deque


class ADWIN:
    """Simplified ADWIN-1 implementation."""

    def __init__(self, delta: float = 0.002):
        """delta = false-positive-rate (smaller = stricter)."""
        self.delta = delta
        self.window: deque[float] = deque()
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        self.drift_detected = False

    def update(self, value: float) -> bool:
        """Add a value. Returns True if drift was detected (oldest data dropped)."""
        self.window.append(value)
        self.total += value
        self.width += 1
        # Update sample variance via Welford
        # (simplified — using O(n) recompute periodically)
        if self.width % 32 == 0:
            arr = list(self.window)
            mu = sum(arr) / len(arr)
            self.variance = sum((x - mu) ** 2 for x in arr) / len(arr)

        return self._check_drift()

    def _check_drift(self) -> bool:
        if self.width < 16:
            return False
        # Search for a split point where the means differ by more than ε
        arr = list(self.window)
        n = len(arr)
        cum = 0.0
        for i in range(1, n - 1):
            cum += arr[i - 1]
            n0 = i
            n1 = n - i
            mu0 = cum / n0
            mu1 = (self.total - cum) / n1
            # Hoeffding bound
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            denom = max(self.variance, 1e-9)
            eps_cut = math.sqrt(2 * denom * math.log(2 * n / self.delta) / m) + (
                2.0 / 3 * math.log(2 * n / self.delta) / m
            )
            if abs(mu0 - mu1) > eps_cut:
                # Drop older half
                drop_n = n0
                while drop_n > 0:
                    val = self.window.popleft()
                    self.total -= val
                    self.width -= 1
                    drop_n -= 1
                self.drift_detected = True
                return True
        return False


__all__ = ["ADWIN"]
