"""
download_vendors.py — Download all vendor JS libs for offline use.

Usage:
  python docs/architecture/system_map/assets/vendor/download_vendors.py

Run once. Requires internet access. Creates/overwrites files in this directory.
"""
import urllib.request
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).parent

LIBS = [
    ("cytoscape.min.js",              "https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"),
    ("cytoscape-fcose.js",            "https://unpkg.com/cytoscape-fcose@2.2.0/cytoscape-fcose.js"),
    ("cytoscape-expand-collapse.js",  "https://unpkg.com/cytoscape-expand-collapse@4.1.0/cytoscape-expand-collapse.js"),
    # cytoscape-navigator: download from https://github.com/cytoscape/cytoscape.js-navigator/releases
    # Not on unpkg — place built UMD file as cytoscape-navigator.js manually if needed
    # ("cytoscape-navigator.js", "..."),
    ("popper.min.js",                 "https://unpkg.com/@popperjs/core@2.11.8/dist/umd/popper.min.js"),
    ("tippy.umd.min.js",              "https://unpkg.com/tippy.js@6.3.7/dist/tippy-bundle.umd.min.js"),
    ("fuse.min.js",                   "https://unpkg.com/fuse.js@7.0.0/dist/fuse.min.js"),
]


def download() -> int:
    errors = 0
    for filename, url in LIBS:
        dest = VENDOR_DIR / filename
        print(f"  Downloading {filename}...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            size = dest.stat().st_size
            print(f"OK ({size // 1024} KB)")
        except Exception as e:
            print(f"FAILED: {e}")
            errors += 1
    if errors:
        print(f"\n[WARN] {errors} download(s) failed. Retry or download manually.")
        return 1
    print("\n[OK] All vendor libs downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(download())
