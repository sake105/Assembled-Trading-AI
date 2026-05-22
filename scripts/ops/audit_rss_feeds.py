"""RSS feed health audit (2026-05-19 external-data audit).

Probes every feed in ``configs/intel/rss_feeds.yaml`` and classifies each
as OK / FAIL / SLOW / EMPTY. Writes structured report to ``output/qa/`` and
prints dead feeds that should be considered for removal.

Conservative: read-only audit, does NOT modify the YAML. Operator decides
what to prune based on the report.

Usage::

    python -m scripts.ops.audit_rss_feeds
    python -m scripts.ops.audit_rss_feeds --timeout 10 --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_FEEDS_YAML = _REPO_ROOT / "configs" / "intel" / "rss_feeds.yaml"
_OUTPUT_DIR = _REPO_ROOT / "output" / "qa"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _classify(url: str, timeout: float) -> dict:
    start = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            body = resp.read(4096)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        ok_status = 200 <= status < 400
        looks_xml = (
            b"<?xml" in body[:200] or b"<rss" in body[:500] or b"<feed" in body[:500]
        )
        if not ok_status:
            return {
                "verdict": "FAIL",
                "status": status,
                "elapsed_ms": elapsed_ms,
                "reason": f"HTTP {status}",
                "content_type": content_type,
            }
        if not looks_xml:
            return {
                "verdict": "EMPTY",
                "status": status,
                "elapsed_ms": elapsed_ms,
                "reason": "no XML/RSS markers in first 4KB",
                "content_type": content_type,
            }
        if elapsed_ms > 4000.0:
            return {
                "verdict": "SLOW",
                "status": status,
                "elapsed_ms": elapsed_ms,
                "reason": f"{elapsed_ms:.0f}ms > 4000ms threshold",
                "content_type": content_type,
            }
        return {
            "verdict": "OK",
            "status": status,
            "elapsed_ms": elapsed_ms,
            "reason": None,
            "content_type": content_type,
        }
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "verdict": "FAIL",
            "status": e.code,
            "elapsed_ms": elapsed_ms,
            "reason": f"HTTPError {e.code}",
            "content_type": None,
        }
    except urllib.error.URLError as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "verdict": "FAIL",
            "status": None,
            "elapsed_ms": elapsed_ms,
            "reason": f"URLError {type(e.reason).__name__ if e.reason else 'unknown'}",
            "content_type": None,
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "verdict": "FAIL",
            "status": None,
            "elapsed_ms": elapsed_ms,
            "reason": f"{type(e).__name__}: {str(e)[:80]}",
            "content_type": None,
        }


def _audit_one(item: dict, timeout: float) -> dict:
    name = item.get("name", "?")
    url = item.get("url", "")
    res = _classify(url, timeout)
    res["name"] = name
    res["url"] = url
    res["category"] = item.get("category")
    return res


def run(timeout: float, workers: int) -> dict:
    with _FEEDS_YAML.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    feeds = cfg["feeds"] if isinstance(cfg, dict) and "feeds" in cfg else cfg
    print(f"[START] auditing {len(feeds)} feeds, timeout={timeout}s, workers={workers}")

    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_audit_one, item, timeout): item for item in feeds}
        for fut in cf.as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                item = futures[fut]
                results.append(
                    {
                        "name": item.get("name", "?"),
                        "url": item.get("url", ""),
                        "verdict": "FAIL",
                        "reason": f"executor-error: {e}",
                    }
                )

    by_verdict: dict[str, list[dict]] = {"OK": [], "SLOW": [], "EMPTY": [], "FAIL": []}
    for r in results:
        by_verdict.setdefault(r["verdict"], []).append(r)

    n_total = len(results)
    n_ok = len(by_verdict["OK"])
    n_slow = len(by_verdict["SLOW"])
    n_empty = len(by_verdict["EMPTY"])
    n_fail = len(by_verdict["FAIL"])

    print()
    print(f"[OK]    {n_ok:>3} / {n_total}")
    print(f"[SLOW]  {n_slow:>3} / {n_total}  (> 4000 ms)")
    print(f"[EMPTY] {n_empty:>3} / {n_total}  (HTTP 200 but no XML markers)")
    print(f"[FAIL]  {n_fail:>3} / {n_total}")

    if by_verdict["FAIL"]:
        print("\nFAIL details:")
        for r in sorted(by_verdict["FAIL"], key=lambda x: x.get("name", "")):
            print(f"  - {r['name']:40s}  {r.get('reason', '?')}  {r.get('url')}")

    if by_verdict["EMPTY"]:
        print("\nEMPTY details (HTTP 200 but no feed content):")
        for r in sorted(by_verdict["EMPTY"], key=lambda x: x.get("name", "")):
            print(f"  - {r['name']:40s}  ct={r.get('content_type')}  {r.get('url')}")

    if by_verdict["SLOW"]:
        print("\nSLOW (consider keeping but with longer timeout):")
        for r in sorted(
            by_verdict["SLOW"], key=lambda x: x.get("elapsed_ms", 0), reverse=True
        ):
            print(f"  - {r['name']:40s}  {r.get('elapsed_ms'):.0f}ms  {r.get('url')}")

    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "n_slow": n_slow,
        "n_empty": n_empty,
        "n_fail": n_fail,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RSS feed health audit")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path for the report JSON",
    )
    args = parser.parse_args()

    report = run(timeout=args.timeout, workers=args.workers)
    out_path = args.output or (_OUTPUT_DIR / "rss_feed_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[OK] wrote report -> {out_path}")
    return 0 if report["n_fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
