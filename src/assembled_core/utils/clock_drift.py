# src/assembled_core/utils/clock_drift.py
"""NTP clock-drift detector (audit C4-043).

Production-grade systems should run ``chrony`` or ``systemd-timesyncd``
and rely on the OS to keep the wall clock disciplined. This helper is
a *secondary*, application-level safety net: it asks a public NTP server
what time it thinks it is, compares to ``datetime.now(UTC)``, and
returns the delta in seconds. Operators wire it into ``/ready`` (or a
heartbeat) so a divergent local clock fails loud instead of silently
producing timestamp-correlated bugs.

The implementation is stdlib-only (no ``ntplib`` dependency): it speaks
the NTPv3 client-mode protocol over UDP with a 2-second socket timeout.
A network failure is reported as ``None`` so callers can decide whether
unknown-drift is degraded or hard-fail.

Example::

    from src.assembled_core.utils.clock_drift import measure_drift_seconds

    drift = measure_drift_seconds()
    if drift is not None and abs(drift) > 0.1:
        logger.warning("clock drift %.3fs — investigate NTP daemon", drift)
"""

from __future__ import annotations

import logging
import socket
import struct
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# NTP epoch is 1900-01-01; UNIX epoch is 1970-01-01 — 70 years in seconds.
_NTP_DELTA = 2_208_988_800
_DEFAULT_HOST = "pool.ntp.org"
_NTP_PORT = 123
_DEFAULT_TIMEOUT_SEC = 2.0


def measure_drift_seconds(
    *,
    host: str = _DEFAULT_HOST,
    timeout: float = _DEFAULT_TIMEOUT_SEC,
) -> float | None:
    """Return ``local_now - ntp_now`` in seconds, or ``None`` on failure.

    Positive value → local clock is **ahead** of NTP. Negative → behind.
    ``None`` → could not reach the server or got an invalid response.

    Args:
        host: NTP server hostname.
        timeout: UDP read timeout in seconds.

    Returns:
        Drift in seconds (float) or ``None``.
    """
    # NTPv3 client packet: LI=0, VN=3, Mode=3 → first byte = 0b00_011_011 = 0x1B
    request = b"\x1b" + b"\x00" * 47

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        local_before = time.time()
        try:
            sock.sendto(request, (host, _NTP_PORT))
            data, _ = sock.recvfrom(48)
        except (socket.gaierror, socket.timeout, OSError) as exc:
            logger.warning("[clock_drift] NTP query failed for %s: %s", host, exc)
            return None
        local_after = time.time()
    finally:
        sock.close()

    if len(data) < 48:
        logger.warning("[clock_drift] short NTP response: %d bytes", len(data))
        return None

    # Server's transmit timestamp = bytes 40..47 (seconds + fraction).
    seconds, fraction = struct.unpack("!II", data[40:48])
    if seconds == 0:
        # Server returned all-zero transmit timestamp — not usable.
        return None
    ntp_unix = seconds - _NTP_DELTA + fraction / 2**32

    # Estimate the local time at the moment the server's response was sent
    # by averaging the local-clock before/after readings (Cristian-style).
    local_mid = (local_before + local_after) / 2.0
    drift = local_mid - ntp_unix
    return float(drift)


def drift_status(
    *,
    host: str = _DEFAULT_HOST,
    warn_seconds: float = 0.1,
    fail_seconds: float = 1.0,
) -> dict[str, object]:
    """Wrapper that classifies drift into ``ok`` / ``warn`` / ``fail`` / ``unknown``.

    Returns a dict suitable for embedding in a readiness probe response::

        {"status": "ok" | "warn" | "fail" | "unknown",
         "drift_seconds": float | None,
         "host": str,
         "checked_at": iso8601-string,
         "warn_threshold": float,
         "fail_threshold": float}
    """
    drift = measure_drift_seconds(host=host)
    checked_at = datetime.now(timezone.utc).isoformat()
    if drift is None:
        status = "unknown"
    elif abs(drift) >= fail_seconds:
        status = "fail"
    elif abs(drift) >= warn_seconds:
        status = "warn"
    else:
        status = "ok"
    return {
        "status": status,
        "drift_seconds": drift,
        "host": host,
        "checked_at": checked_at,
        "warn_threshold": warn_seconds,
        "fail_threshold": fail_seconds,
    }
