"""Generate performance profile report from profiling outputs.

This script collects timing and profiling data from output/profiles/ and
generates a Markdown report (docs/PERFORMANCE_PROFILE.md) with:
- Runtime summaries per job
- Step-by-step timing breakdowns
- Top-3 performance hotspots per job
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

try:
    import pstats
except ImportError:
    pstats = None  # type: ignore[assignment,unused-ignore]

logger = logging.getLogger(__name__)


def find_latest_job_run(profiles_root: Path, job_name: str) -> Path | None:
    """Find the latest run directory for a job.

    Args:
        profiles_root: Root directory containing job profiles (e.g., output/profiles/)
        job_name: Name of the job (e.g., "BACKTEST_MEDIUM")

    Returns:
        Path to the latest run directory, or None if not found
    """
    job_dir = profiles_root / job_name
    if not job_dir.exists():
        return None

    # Find all timestamped subdirectories
    run_dirs = [d for d in job_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None

    # Sort by directory name (timestamp should sort lexicographically)
    run_dirs_sorted = sorted(run_dirs, key=lambda p: p.name, reverse=True)
    return run_dirs_sorted[0]


def load_timings_json(timings_path: Path) -> dict[str, Any] | None:
    """Load timings.json file.

    Args:
        timings_path: Path to timings.json file

    Returns:
        Dictionary with timing data, or None if file doesn't exist
    """
    if not timings_path.exists():
        return None

    try:
        with open(timings_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load timings.json from {timings_path}: {e}")
        return None


def extract_top_hotspots_from_pstats(
    pstats_path: Path, top_n: int = 3
) -> list[dict[str, Any]]:
    """Extract top hotspots from cProfile .prof file.

    Args:
        pstats_path: Path to .prof file
        top_n: Number of top hotspots to extract (default: 3)

    Returns:
        List of dictionaries with keys: function, file, line, cumtime, ncalls
    """
    if pstats is None:
        logger.warning("pstats module not available, cannot extract hotspots")
        return []

    if not pstats_path.exists():
        return []

    try:
        stats = pstats.Stats(str(pstats_path))
        stats.sort_stats(pstats.SortKey.CUMULATIVE)

        hotspots = []
        # stats.stats is a dict where keys are (filename, line_number, function_name) tuples
        # and values are tuples: (primitive_call_count, total_call_count, total_time, cumulative_time, callers_dict)
        # sort_stats() modifies the order internally, but stats.stats.items() still returns unsorted.
        # We need to manually sort by cumulative_time (index 3 in value tuple) and take top_n.
        stats_list = list(stats.stats.items())
        # Sort by cumulative time (ct, which is index 3 in the value tuple), descending
        stats_list_sorted = sorted(stats_list, key=lambda x: x[1][3], reverse=True)
        
        for func_tuple, stats_data in stats_list_sorted[:top_n]:
            # func_tuple is (filename, line_number, function_name)
            file_path, line_num, func_name = func_tuple
            # stats_data is (primitive_calls, total_calls, total_time, cumulative_time, callers)
            cc, nc, tt, ct, callers = stats_data

            hotspots.append(
                {
                    "function": func_name,
                    "file": str(file_path),
                    "line": line_num,
                    "cumtime": ct,
                    "ncalls": nc,
                }
            )

        return hotspots
    except Exception as e:
        logger.warning(f"Failed to extract hotspots from {pstats_path}: {e}")
        return []


def extract_runtime_from_timings(timings_data: dict[str, Any]) -> float | None:
    """Extract total runtime from timings data.

    Args:
        timings_data: Dictionary loaded from timings.json

    Returns:
        Total runtime in seconds, or None if not available
    """
    if "summary" in timings_data and "total_duration_ms" in timings_data["summary"]:
        return timings_data["summary"]["total_duration_ms"] / 1000.0

    # Fallback: sum all step durations
    if "steps" in timings_data:
        total_ms = sum(
            step.get("duration_ms", 0) for step in timings_data["steps"].values()
        )
        return total_ms / 1000.0

    return None


def format_step_breakdown(timings_data: dict[str, Any]) -> str:
    """Format step-by-step timing breakdown as Markdown.

    Args:
        timings_data: Dictionary loaded from timings.json

    Returns:
        Markdown-formatted string with step breakdown
    """
    if "steps" not in timings_data:
        return "No step timing data available."

    steps = timings_data["steps"]
    if not steps:
        return "No steps recorded."

    lines = []
    lines.append("| Step | Duration (ms) | Duration (s) |")
    lines.append("|------|---------------|--------------|")

    # Sort steps by duration (descending)
    sorted_steps = sorted(
        steps.items(), key=lambda x: x[1].get("duration_ms", 0), reverse=True
    )

    for step_name, step_data in sorted_steps:
        duration_ms = step_data.get("duration_ms", 0)
        duration_s = duration_ms / 1000.0
        lines.append(f"| {step_name} | {duration_ms:.2f} | {duration_s:.4f} |")

    return "\n".join(lines)


def format_hotspots(hotspots: list[dict[str, Any]]) -> str:
    """Format hotspots as Markdown table.

    Args:
        hotspots: List of hotspot dictionaries

    Returns:
        Markdown-formatted string with hotspots table
    """
    if not hotspots:
        return "No hotspot data available."

    lines = []
    lines.append("| Rank | Function | File | Line | Cum Time (s) | Calls |")
    lines.append("|------|----------|------|------|--------------|-------|")

    for idx, hotspot in enumerate(hotspots, start=1):
        func_name = hotspot.get("function", "?")
        file_path = hotspot.get("file", "?")
        # Extract just the filename for readability
        file_name = Path(file_path).name if file_path != "?" else "?"
        line_num = hotspot.get("line", "?")
        cumtime = hotspot.get("cumtime", 0.0)
        ncalls = hotspot.get("ncalls", 0)

        lines.append(f"| {idx} | `{func_name}` | `{file_name}` | {line_num} | {cumtime:.4f} | {ncalls} |")

    return "\n".join(lines)


def generate_report_markdown(
    jobs_data: dict[str, dict[str, Any]], output_path: Path
) -> None:
    """Generate Markdown report from jobs data.

    Args:
        jobs_data: Dictionary mapping job names to their data
        output_path: Path to write the report
    """
    lines = []
    lines.append("# Performance Profile Report")
    lines.append("")
    lines.append(
        "This report summarizes performance characteristics of reference jobs "
        "based on profiling data."
    )
    lines.append("")
    lines.append("**Generated by:** `scripts/generate_performance_profile_report.py`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append("| Job | Runtime (s) | Latest Run |")
    lines.append("|-----|-------------|------------|")

    for job_name in sorted(jobs_data.keys()):
        job_info = jobs_data[job_name]
        runtime = job_info.get("runtime", "N/A")
        latest_run = job_info.get("latest_run", "N/A")
        lines.append(f"| {job_name} | {runtime} | {latest_run} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-job details
    for job_name in sorted(jobs_data.keys()):
        job_info = jobs_data[job_name]
        lines.append(f"## {job_name}")
        lines.append("")

        # Runtime
        runtime = job_info.get("runtime")
        if runtime is not None:
            lines.append(f"**Total Runtime:** {runtime:.4f} seconds")
        else:
            lines.append("**Total Runtime:** N/A")
        
        # Cache warm indicator
        cache_warm = job_info.get("cache_warm", False)
        if cache_warm:
            lines.append("")
            lines.append("> ⚡ **Cache warm**: This run used pre-computed factors from the Factor Store (cache hit).")
        
        lines.append("")

        # Step breakdown
        step_breakdown = job_info.get("step_breakdown")
        if step_breakdown:
            lines.append("### Step Breakdown")
            lines.append("")
            lines.append(step_breakdown)
            lines.append("")

        # Hotspots
        hotspots = job_info.get("hotspots", [])
        if hotspots:
            lines.append("### Top-3 Performance Hotspots")
            lines.append("")
            lines.append(format_hotspots(hotspots))
            lines.append("")
        else:
            lines.append("### Performance Hotspots")
            lines.append("")
            lines.append("No hotspot data available.")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Performance profile report written to {output_path}")


def generate_performance_profile_report(
    profiles_root: Path, output_path: Path, job_names: list[str] | None = None
) -> None:
    """Generate performance profile report from profiling outputs.

    Args:
        profiles_root: Root directory containing job profiles (e.g., output/profiles/)
        output_path: Path to write the report (e.g., docs/PERFORMANCE_PROFILE.md)
        job_names: Optional list of job names to include (default: all found jobs)
    """
    if not profiles_root.exists():
        logger.warning(f"Profiles root directory does not exist: {profiles_root}")
        logger.info("Creating report with placeholder structure...")
        # Create placeholder report
        placeholder_data = {
            "EOD_SMALL": {
                "runtime": None,
                "latest_run": "N/A",
                "step_breakdown": "No data available.",
                "hotspots": [],
            },
            "BACKTEST_MEDIUM": {
                "runtime": None,
                "latest_run": "N/A",
                "step_breakdown": "No data available.",
                "hotspots": [],
            },
            "ML_JOB": {
                "runtime": None,
                "latest_run": "N/A",
                "step_breakdown": "No data available.",
                "hotspots": [],
            },
        }
        generate_report_markdown(placeholder_data, output_path)
        return

    # Determine which jobs to process
    if job_names is None:
        # Find all job directories
        job_dirs = [d for d in profiles_root.iterdir() if d.is_dir()]
        job_names = [d.name for d in job_dirs]

    jobs_data: dict[str, dict[str, Any]] = {}

    for job_name in job_names:
        logger.info(f"Processing job: {job_name}")

        latest_run_dir = find_latest_job_run(profiles_root, job_name)
        if not latest_run_dir:
            logger.warning(f"No run found for job {job_name}")
            jobs_data[job_name] = {
                "runtime": None,
                "latest_run": "N/A",
                "step_breakdown": "No run data available.",
                "hotspots": [],
            }
            continue

        logger.info(f"  Latest run: {latest_run_dir.name}")

        # Load timings.json
        # For BACKTEST_MEDIUM, timings.json is written to profile_dir/timings.json by profile_jobs.py
        timings_path = latest_run_dir / "timings.json"
        # Also check parent directory (some setups might have timings at job level)
        if not timings_path.exists():
            timings_path = latest_run_dir.parent.parent / "timings.json"

        timings_data = load_timings_json(timings_path) if timings_path.exists() else None
        
        # Load profile_summary.log to check for cache_warm flag
        profile_summary_path = latest_run_dir / "profile_summary.log"
        cache_warm = False
        if profile_summary_path.exists():
            try:
                with open(profile_summary_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("cache_warm="):
                            cache_warm = line.strip().split("=", 1)[1].lower() == "true"
                            break
            except Exception as e:
                logger.warning(f"Failed to read profile_summary.log from {profile_summary_path}: {e}")

        # Extract runtime
        runtime = None
        step_breakdown = None
        if timings_data:
            runtime = extract_runtime_from_timings(timings_data)
            step_breakdown = format_step_breakdown(timings_data)

        # Find and load pstats file
        pstats_path = latest_run_dir / f"profile_{job_name}.prof"
        hotspots = []
        if pstats_path.exists():
            hotspots = extract_top_hotspots_from_pstats(pstats_path, top_n=3)
        else:
            # Try alternative naming
            prof_files = list(latest_run_dir.glob("*.prof"))
            if prof_files:
                hotspots = extract_top_hotspots_from_pstats(prof_files[0], top_n=3)

        jobs_data[job_name] = {
            "runtime": runtime,
            "latest_run": latest_run_dir.name,
            "step_breakdown": step_breakdown or "No step timing data available.",
            "hotspots": hotspots,
            "cache_warm": cache_warm,
        }

    generate_report_markdown(jobs_data, output_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate performance profile report from profiling outputs."
    )
    parser.add_argument(
        "--profiles-root",
        type=Path,
        default=Path("output/profiles"),
        help="Root directory containing job profiles (default: output/profiles)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/PERFORMANCE_PROFILE.md"),
        help="Path to write the report (default: docs/PERFORMANCE_PROFILE.md)",
    )
    parser.add_argument(
        "--jobs",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of job names to include (default: all found jobs)",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    try:
        generate_performance_profile_report(
            profiles_root=args.profiles_root,
            output_path=args.output,
            job_names=args.jobs,
        )
        return 0
    except Exception as e:
        logger.error(f"Failed to generate performance profile report: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

