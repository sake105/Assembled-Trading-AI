# External Services Setup Runbook

> Audit C2-011 + I-002 + I-003 + alerting channels — the system relies
> on several **external** services for full operational maturity.
> None of these can be set up from the codebase autonomously (they
> need an account, a billing entry, an API key). This runbook turns
> each into a 5-minute "you can copy-paste this" procedure.
>
> Order below is roughly priority: dead-man-switch first
> because it catches silent outages, B2 + Litestream next for DR,
> rest is optional polish.

---

## 0. Master checklist

- [ ] **Dead-man-switch (Uptime-Robot or healthchecks.io)** → cron heartbeat
- [ ] **Backblaze B2** bucket with Object Lock → audit-log offsite
- [ ] **Litestream** sidecar for SQLite continuous replication
- [ ] **Telegram bot** (optional) → TELEGRAM_BOT_TOKEN
- [ ] **Email SMTP** (optional, last-resort fallback) → SMTP_HOST/USER/PASS
- [ ] **Cloudflare DNS** (only if you go multi-region — out of scope here)

After every check, verify by triggering the corresponding alert path
from the codebase (a kill-switch activate, a reconciliation fail
simulation) and confirming receipt.

---


## 1. Dead-man-switch (heartbeat monitor)

**Why:** detects when the local host has crashed silently — the
codebase can't alert if the process is dead. An external service
expects a periodic ping and alerts WHEN IT STOPS arriving.

**Recommended: healthchecks.io** (free tier covers 1 monitor).

**Setup (5 min):**

1. https://healthchecks.io/ → sign up, free tier.
2. Create a check named `assembled-trading-heartbeat`. Set:
   - Period: `1 hour`
   - Grace: `15 min` (covers brief restarts)
3. Copy the ping URL (e.g. `https://hc-ping.com/<uuid>`).
4. Add a cron job (Linux) or Windows Task Scheduler entry:
   ```bash
   # Cron, every 30 minutes:
   */30 * * * * /usr/bin/curl -fsS --retry 3 https://hc-ping.com/<uuid> > /dev/null
   ```
   ```powershell
   # Windows Task Scheduler, every 30 min:
   schtasks /Create /SC MINUTE /MO 30 /TN ATAI-Heartbeat ^
     /TR "curl.exe -fsS --retry 3 https://hc-ping.com/<uuid>"
   ```
5. Set up email/Telegram notification *on healthchecks.io* so the
   operator hears it within 15 min of the heartbeat stopping.

**Cost:** free.

**Alternative:** **Uptime-Robot** free tier (50 monitors). Use HTTP
GET to a fictional internal endpoint that returns 200 only when
the system is healthy — but Uptime-Robot doesn't speak "dead-man"
naturally, healthchecks.io is the better fit.

---

## 2. Backblaze B2 with Object Lock

**Why:** off-site WORM-compliant storage for audit logs (audit
C3-074 + GoBD WORM policy). Object Lock in Compliance mode means
the operator themselves cannot delete a record until the lock
expires — exactly what the regulator wants.

**Setup (15 min):**

1. https://www.backblaze.com/cloud-storage → sign up (free tier:
   10 GB free, $0.005 / GB-month after).
2. Create a bucket `assembled-trading-audit-<unique-suffix>`,
   region `eu-central-003`, type **Private**.
3. Enable Object Lock during creation. Set default retention to
   `3650 days` (10 years), mode `Compliance`.
4. Create an Application Key with `writeFiles + readFiles` scope
   ONLY on this bucket (not master key).
5. Save the keyID + applicationKey somewhere safe; set in shell:
   ```
   B2_KEY_ID=...
   B2_APP_KEY=...
   B2_BUCKET=assembled-trading-audit-<suffix>
   ```
6. Daily replication via `rclone` (cross-platform):
   ```bash
   rclone config  # interactive — pick Backblaze, paste creds
   rclone copy output/ops/ remote:assembled-trading-audit-<suffix>/ops/ \
     --include "*.jsonl" --transfers 4
   ```
7. Add to cron / Task Scheduler:
   ```
   # daily at 23:30
   30 23 * * * /usr/bin/rclone copy output/ops/ remote:...
   ```

**Cost:** ~$0.05 / month on hobby scale (10 GB of audit logs is a
LOT of trading activity).

A scaffold script is at `scripts/ops/setup_b2_backup.sh` (Wave-10
deliverable).

---

## 3. Litestream — SQLite continuous replication

**Why:** the paper-trading ledger lives in `data/paper_ledger.db`
(SQLite). Litestream replicates SQLite to S3/B2 at second-level
granularity. Audit I-002.

**Setup (10 min):**

1. Download Litestream binary:
   - Linux: `wget https://github.com/benbjohnson/litestream/releases/...`
   - Windows: download .exe, place in `C:\Tools\litestream\`.
2. Create `/etc/litestream.yml` (Linux) or
   `C:\Tools\litestream\litestream.yml` (Windows). See the
   scaffold at `configs/integrations/litestream.yml.example`.
3. Run as service:
   - Linux: `systemctl enable --now litestream.service`
   - Windows: `nssm install litestream`
4. **Verify replication after 1 min:**
   ```bash
   litestream snapshots data/paper_ledger.db
   ```
   Should list at least 1 snapshot.

**Cost:** the binary is free; replication target is the B2 bucket
from §3.

---

## 4. Telegram bot

**Why:** Primary push-notification channel for AlertManager rules. Already supported by `AlertManager` channel kind `telegram`.

**Setup (5 min):**

1. In Telegram, message `@BotFather`. Send `/newbot`. Pick a name
   and handle. Save the token (looks like `123456:ABCDE-...`).
2. Send a message to your new bot from your own Telegram account.
3. Fetch your chat-id by visiting
   `https://api.telegram.org/bot<TOKEN>/getUpdates`. Look for
   `chat.id` in the JSON.
4. Set env:
   ```
   TELEGRAM_BOT_TOKEN=123456:ABCDE-...
   TELEGRAM_CHAT_ID=<id>
   ```
5. Verify:
   ```bash
   python -c "from src.assembled_core.ops.alerting import AlertManager; AlertManager()._send_telegram({}, 'test')"
   ```

**Cost:** free.

---

## 5. Email SMTP (last-resort fallback)

**Why:** last-resort fallback when Telegram is unavailable. Used by
`AlertManager` channel kind `email`.

**Setup (5 min):**

1. Get SMTP credentials from your email provider (Gmail app-password,
   Mailgun, SendGrid, your own ISP, etc.).
2. Set env:
   ```
   SMTP_HOST=smtp.gmail.com
   SMTP_USER=...
   SMTP_PASS=...
   ALERT_EMAIL_TO=hans+alerts@<domain>
   ```
3. Verify:
   ```bash
   python -c "from src.assembled_core.ops.alerting import AlertManager; AlertManager()._send_email({}, 'test', 'body')"
   ```

**Cost:** free if you use Gmail / outlook.

---

## 7. Multi-region DNS failover (advanced, out of scope today)

If you go multi-region later (audit I-008), Cloudflare DNS Load
Balancing with health checks is the standard pick. Out of scope
for the current single-Hetzner deployment.

---

## What this runbook is NOT

- Not a substitute for setting up your own monitoring stack
  (Prometheus / Grafana / Loki) — those are bigger investments and
  audit F-001 covers them separately.
- Not a backup-restore drill. Wiring B2 is half the work; *testing
  that you can actually restore from it* is the other half — see
  audit I-007 and `docs/OPERATOR_RUNBOOK.md` Quarterly DR drill.
- Not zero-cost: while each individual service is free / nearly-free
  on hobby tier, the total adds up. Re-evaluate at $20/month
  threshold.
