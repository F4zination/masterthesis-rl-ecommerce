"""Generate the WebP product-image variants the demo shops actually serve.

The source PNGs are 1254x1254 and average ~1.9 MB each. Serving them
directly meant ~40 MB per category page, which is what made the shop feel
slow. This script renders two derivatives per product:

    thumb/  420px q72  -- listing grids, cart rows, recommendation widgets
    full/  1100px q82  -- the product detail hero

Both are committed to the repository and shipped in place of the originals
(see the ``.dockerignore`` entries), so the containers carry tens of
megabytes instead of half a gigabyte and no image tooling is needed at
build time.

    python tools/build_image_variants.py           # generate what's stale
    python tools/build_image_variants.py --force   # re-render everything
    python tools/build_image_variants.py --check   # verify, write nothing

``--check`` exits non-zero if any variant is missing or older than its
source, which is what the per-site test suites call.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SITES = ("DemoSiteV2", "DemoSiteV3")

# (directory name, longest edge in px, WebP quality)
VARIANTS = (
    ("thumb", 420, 72),
    ("full", 1100, 82),
)


def _sources(site: str) -> list[Path]:
    root = REPO_ROOT / site / "app" / "static" / "images" / "products"
    return sorted(p for p in root.rglob("*.png") if p.is_file())


def _target(site: str, source: Path, variant: str) -> Path:
    """Map a source PNG to its variant path, mirroring the category layout."""
    products = REPO_ROOT / site / "app" / "static" / "images" / "products"
    rel = source.relative_to(products).with_suffix(".webp")
    return REPO_ROOT / site / "app" / "static" / "images" / variant / rel


def _stale(source: Path, target: Path) -> bool:
    return not target.exists() or target.stat().st_mtime < source.stat().st_mtime


def _render(source: Path, target: Path, edge: int, quality: int) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        im = im.convert("RGB")
        im.thumbnail((edge, edge), Image.LANCZOS)
        # method=6 is the slowest/densest WebP search; worth it because these
        # are built once and then served for the lifetime of the study.
        im.save(target, "WEBP", quality=quality, method=6)
    return target.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-render up-to-date variants")
    ap.add_argument("--check", action="store_true", help="verify only; write nothing")
    args = ap.parse_args()

    missing: list[str] = []
    jobs: list[tuple[Path, Path, int, int]] = []
    src_bytes = out_bytes = 0

    for site in SITES:
        sources = _sources(site)
        if not sources:
            print(f"!! {site}: no source images found", file=sys.stderr)
            return 1
        for source in sources:
            src_bytes += source.stat().st_size
            for variant, edge, quality in VARIANTS:
                target = _target(site, source, variant)
                if _stale(source, target):
                    if args.check:
                        missing.append(str(target.relative_to(REPO_ROOT)))
                    else:
                        jobs.append((target, source, edge, quality))
                elif not args.check:
                    out_bytes += target.stat().st_size

    if args.check:
        if missing:
            print(f"{len(missing)} variant(s) missing or stale; run "
                  f"`python tools/build_image_variants.py`:", file=sys.stderr)
            for m in missing[:20]:
                print(f"  {m}", file=sys.stderr)
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more", file=sys.stderr)
            return 1
        print("All image variants are present and up to date.")
        return 0

    if args.force:
        jobs = [
            (_target(s, p, v), p, e, q)
            for s in SITES for p in _sources(s) for v, e, q in VARIANTS
        ]
        out_bytes = 0

    if not jobs:
        print("Nothing to do; all variants are up to date.")
        return 0

    print(f"Rendering {len(jobs)} variant(s)...")
    with ThreadPoolExecutor() as pool:
        results = pool.map(lambda j: _render(j[1], j[0], j[2], j[3]), jobs)
        out_bytes += sum(results)

    mb = 1024 * 1024
    print(f"Done. Sources {src_bytes / mb:.0f} MB -> variants {out_bytes / mb:.0f} MB "
          f"({out_bytes / src_bytes:.1%} of original weight).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
