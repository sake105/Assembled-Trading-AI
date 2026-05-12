#!/usr/bin/env bash
# scripts/ops/setup_uptime_robot.sh
# Audit C2-011 — dead-man-switch heartbeat via healthchecks.io.
#
# Sends a single ping to a healthchecks.io URL. Wire this into cron
# (Linux) or Task Scheduler (Windows) at 30-min intervals.
#
# See docs/EXTERNAL_SERVICES_SETUP.md §2.

set -euo pipefail

: "${HEALTHCHECKS_URL:?HEALTHCHECKS_URL env required (https://hc-ping.com/<uuid>)}"

# Best-effort ping; never fail the cron job because of network blips.
curl -fsS --max-time 10 --retry 3 --retry-delay 5 "${HEALTHCHECKS_URL}" > /dev/null || true

# Optional: include a local healthcheck before pinging so we only signal
# "alive" when the trading process is also responsive. Uncomment to use:
#
# if ! curl -fsS --max-time 5 "http://localhost:8000/live" > /dev/null; then
#     echo "[heartbeat] local /live failed — skipping external ping"
#     exit 1
# fi
# curl -fsS --max-time 10 "${HEALTHCHECKS_URL}" > /dev/null

exit 0
