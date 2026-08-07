"""Product image integrity: every seeded image must resolve on Linux.

Runnable standalone (no pytest required):

    .venv/Scripts/python DemoSiteV2/tests/test_product_images.py

This guards a class of bug that is invisible during local development on
Windows/macOS: those filesystems are case-insensitive, so a seed entry
referencing ``clothing/Hoodie.png`` happily resolves a file that git tracks
as ``Clothing/Hoodie.png``. Inside the Linux container it 404s. That is how
every Clothing and Electronics image (99 of 246) silently broke in
production while looking fine on the developer machine.

The check therefore compares against an explicit directory listing rather
than ``os.path.exists``, which would inherit the host filesystem's
case-folding and pass on Windows regardless.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
SEED = APP_DIR / "app" / "seed.py"
IMAGE_ROOT = APP_DIR / "app" / "static" / "images" / "products"


def _seeded_refs() -> list[str]:
    """Return the product-image paths referenced by ``seed.py``.

    Only refs carrying a category directory are returned; the bare filename
    in the ``_local`` docstring is illustrative, not a real product.
    """
    src = SEED.read_text(encoding="utf-8")
    return sorted({r for r in re.findall(r'_local\("([^"]+)"\)', src) if "/" in r})


def _files_on_disk() -> set[str]:
    """Return every product image path, preserving exact on-disk case."""
    return {
        p.relative_to(IMAGE_ROOT).as_posix()
        for p in IMAGE_ROOT.rglob("*")
        if p.is_file()
    }


def test_every_seeded_image_exists_with_exact_case() -> None:
    refs = _seeded_refs()
    actual = _files_on_disk()
    assert refs, f"no image references parsed from {SEED}"

    lowered = {a.lower(): a for a in actual}
    broken = []
    for ref in refs:
        if ref in actual:
            continue
        # Distinguish a case mismatch (the dangerous, silent one) from a
        # genuinely absent file, so the failure message says which it is.
        match = lowered.get(ref.lower())
        broken.append(f"  {ref!r} -> {'tracked as ' + repr(match) if match else 'MISSING'}")

    assert not broken, (
        f"{len(broken)} of {len(refs)} seeded images will 404 on a "
        "case-sensitive filesystem:\n" + "\n".join(broken)
    )


def test_every_seeded_image_has_web_variants() -> None:
    """Both WebP derivatives must exist for every seeded product.

    The shops render only these variants and the source PNGs are excluded
    from the container images, so a missing variant is a hard 404 in
    production rather than merely a slow page. Regenerate with
    ``make images`` after adding or replacing any product image.
    """
    static = APP_DIR / "app" / "static" / "images"
    missing: list[str] = []
    for variant in ("thumb", "full"):
        root = static / variant
        present = {
            p.relative_to(root).as_posix()
            for p in root.rglob("*.webp")
            if p.is_file()
        }
        for ref in _seeded_refs():
            want = ref.rsplit(".", 1)[0] + ".webp"
            if want not in present:
                missing.append(f"  {variant}/{want}")

    assert not missing, (
        f"{len(missing)} image variant(s) missing or misnamed; run "
        "`make images`:\n" + "\n".join(missing[:20])
    )


def test_seeded_database_image_urls_match_disk() -> None:
    """A pre-existing database volume must not serve stale image paths.

    The two checks above validate ``seed.py``, but the shop serves whatever is
    in ``data/demosite.db``. That database is a runtime volume kept across
    restarts, so a volume created before an image path was corrected holds the
    old value until something realigns it. ``reconcile_product_images`` does
    exactly that on every startup, which means a stale row here says the
    instance has not been started since the asset was renamed -- or that
    reconciliation is not doing its job. Both are worth catching before
    recruitment, because the symptom is a silently broken product image that
    differs between the two conditions.

    Skips when no database is present, since the volume is created on first run
    and is deliberately not tracked in git.
    """
    db_path = APP_DIR / "data" / "demosite.db"
    if not db_path.exists():
        return

    import sqlite3

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT id, name, image_url FROM products WHERE image_url IS NOT NULL"
        ).fetchall()
    except sqlite3.DatabaseError:  # not yet migrated
        return
    finally:
        connection.close()

    actual = _files_on_disk()
    prefix = "/static/images/products/"
    stale = []
    for product_id, name, image_url in rows:
        if not image_url.startswith(prefix):
            continue
        ref = image_url[len(prefix):]
        if ref not in actual:
            stale.append(f"  id={product_id} {name!r}: {ref!r}")

    assert not stale, (
        f"{len(stale)} of {len(rows)} product rows in {db_path.name} reference an "
        "image that does not exist on disk. Start the shop (or redeploy) so "
        "reconcile_product_images realigns the volume:\n" + "\n".join(stale[:20])
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}\n{exc}")
    sys.exit(1 if failures else 0)
