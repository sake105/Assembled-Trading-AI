#!/usr/bin/env bash
# scripts/ops/setup_b2_backup.sh
# Audit I-003 + GoBD WORM — guided Backblaze B2 backup setup.
#
# This script does NOT create the B2 account (that needs a browser),
# but it does:
#   1. validate rclone is installed
#   2. probe that the B2 credentials in env reach the bucket
#   3. dry-run the first sync so the operator sees exactly what would copy
#   4. emit a cron line the operator pastes manually
#
# Run from repo root.
set -euo pipefail

err() { echo "[ERR] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }

# 1. Prerequisites
command -v rclone >/dev/null 2>&1 || err "rclone not installed. Install: https://rclone.org/install/"

: "${B2_KEY_ID:?B2_KEY_ID env required — see docs/EXTERNAL_SERVICES_SETUP.md §3}"
: "${B2_APP_KEY:?B2_APP_KEY env required}"
: "${B2_BUCKET:?B2_BUCKET env required}"

REMOTE_NAME="${B2_REMOTE_NAME:-assembled-b2}"
SRC_DIR="${1:-output/ops}"
DEST_PATH="${REMOTE_NAME}:${B2_BUCKET}/${SRC_DIR##*/}"

# 2. Create rclone remote inline via env overrides (no config file written)
info "Configuring rclone remote '${REMOTE_NAME}' for bucket '${B2_BUCKET}'"
UPPER="${REMOTE_NAME^^}"
declare "RCLONE_CONFIG_${UPPER}_TYPE=b2"
declare "RCLONE_CONFIG_${UPPER}_ACCOUNT=${B2_KEY_ID}"
declare "RCLONE_CONFIG_${UPPER}_KEY=${B2_APP_KEY}"
export "RCLONE_CONFIG_${UPPER}_TYPE" "RCLONE_CONFIG_${UPPER}_ACCOUNT" "RCLONE_CONFIG_${UPPER}_KEY"

# 3. Probe — list the bucket. Empty or non-empty is fine; auth failure is not.
info "Probing bucket access..."
rclone lsd "${REMOTE_NAME}:${B2_BUCKET}" >/dev/null \
    || err "Cannot list bucket '${B2_BUCKET}'. Check B2_KEY_ID/B2_APP_KEY/B2_BUCKET and Object Lock setting."
info "  ok."

# 4. Dry-run the first sync.
info "Dry-run: rclone copy ${SRC_DIR} ${DEST_PATH}"
rclone copy "${SRC_DIR}" "${DEST_PATH}" --include "*.jsonl" --transfers 4 --dry-run

cat <<EOF

[NEXT STEPS]

Manual: append to your crontab (Linux) or schtasks (Windows) — every
day at 23:30:

    30 23 * * * cd $(pwd) && B2_KEY_ID=... B2_APP_KEY=... B2_BUCKET=${B2_BUCKET} \\
        bash scripts/ops/setup_b2_backup.sh ${SRC_DIR}

Manual: enable Object Lock on '${B2_BUCKET}' in the B2 web UI with
'Compliance' mode + retention 3650 days (10 years) for GoBD.
See docs/GOBD_WORM_POLICY.md §4.

EOF

# 5. Actual run on user confirmation.
read -p "Run the real sync now? [y/N] " confirm
if [[ "${confirm,,}" == "y" ]]; then
    rclone copy "${SRC_DIR}" "${DEST_PATH}" --include "*.jsonl" --transfers 4 -v
    info "Done. Verify in B2 web UI."
fi
