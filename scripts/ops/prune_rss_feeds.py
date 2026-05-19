"""Apply audit results from output/qa/rss_feed_audit.json to rss_feeds.yaml.

Strategy:
- Remove feeds with verdict in {FAIL with 404, FAIL with gaierror, EMPTY, empty-URL}
  (truly dead, not recoverable with a better client).
- Keep but annotate feeds with verdict in {FAIL with 403, FAIL with SSL, SLOW, timeout}
  (potentially recoverable via better User-Agent / cert handling / longer timeout).
- Leave OK feeds untouched.

Writes a side-by-side preview (configs/intel/rss_feeds.yaml.pruned) for diff-review,
then asks the operator to confirm by renaming. Or pass --apply to overwrite directly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

_AUDIT_TAG_RE = re.compile(r"\s*\[audit-\d{4}-\d{2}-\d{2}[^\]]*\]")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FEEDS_YAML = _REPO_ROOT / "configs" / "intel" / "rss_feeds.yaml"
_AUDIT_JSON = _REPO_ROOT / "output" / "qa" / "rss_feed_audit.json"


def _is_truly_dead(audit_entry: dict) -> bool:
    if not audit_entry.get("url"):
        return True
    verdict = audit_entry.get("verdict")
    reason = audit_entry.get("reason", "") or ""
    content_type = (audit_entry.get("content_type") or "").lower()
    if verdict == "EMPTY":
        # F-senior-4: EMPTY-with-HTML/JSON content-type = publisher served
        # a landing page, not an RSS feed. EMPTY without that signal could
        # be a publisher quirk recoverable via better client — keep it.
        if "html" in content_type or "json" in content_type:
            return True
        return False
    if verdict == "FAIL":
        if "HTTPError 404" in reason:
            return True
        if "gaierror" in reason:
            return True
    return False


def _is_recoverable(audit_entry: dict) -> bool:
    verdict = audit_entry.get("verdict")
    reason = audit_entry.get("reason", "") or ""
    if verdict == "SLOW":
        return True
    if verdict == "FAIL":
        if "HTTPError 403" in reason:
            return True
        if "SSL" in reason or "ssl" in reason:
            return True
        if "timed out" in reason or "TimeoutError" in reason:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply", action="store_true", help="Overwrite the original YAML in place"
    )
    args = parser.parse_args()

    with _AUDIT_JSON.open(encoding="utf-8") as f:
        audit = json.load(f)
    audit_by_name = {r["name"]: r for r in audit["results"]}

    with _FEEDS_YAML.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if isinstance(cfg, dict) and "feeds" in cfg:
        feeds = cfg["feeds"]
    else:
        feeds = cfg
        cfg = {"feeds": feeds}

    kept: list[dict] = []
    pruned: list[tuple[str, str]] = []
    annotated: list[tuple[str, str]] = []

    for item in feeds:
        name = item.get("name", "?")
        a = audit_by_name.get(name)
        if a is None:
            kept.append(item)
            continue
        # Preserve operator-disabled entries verbatim: an explicit `enabled: false`
        # signals "kept for a reason" (test fixture, deferred re-enable, etc.) —
        # the audit should not override that. Same for missing-enabled (assumed
        # true). Only entries where the operator left `enabled: true` are
        # subject to pruning by this script.
        if item.get("enabled", True) is False:
            kept.append(item)
            continue
        if _is_truly_dead(a):
            pruned.append((name, a.get("reason") or a.get("verdict")))
            continue
        if _is_recoverable(a):
            note = a.get("reason") or a.get("verdict")
            item = dict(item)
            # F-senior-3: strip any previous audit-tag before appending so
            # repeated runs converge instead of accumulating.
            existing = _AUDIT_TAG_RE.sub("", item.get("notes", "") or "").strip()
            tag = f"[audit-2026-05-19 {a['verdict']}: {note}]"
            item["notes"] = (existing + " " + tag).strip() if existing else tag
            annotated.append((name, tag))
        kept.append(item)

    print(
        f"[summary] feeds_in={len(feeds)} kept={len(kept)} pruned={len(pruned)} annotated={len(annotated)}"
    )
    print()
    print("PRUNED (truly dead):")
    for n, r in sorted(pruned):
        print(f"  - {n:40s} {r}")
    print()
    print("ANNOTATED (potentially recoverable):")
    for n, t in sorted(annotated):
        print(f"  - {n:40s} {t}")

    out_path = _FEEDS_YAML if args.apply else _FEEDS_YAML.with_suffix(".yaml.pruned")
    cfg["feeds"] = kept
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print()
    print(f"[OK] wrote -> {out_path}")
    if not args.apply:
        print("Run with --apply to overwrite the original.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
