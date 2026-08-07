"""Map canonical product image URLs onto the WebP variants actually served.

``seed.py`` stores the original PNG path for each product (for example
``/static/images/products/clothing/Hoodie.png``). Those originals are
1254x1254 and ~1.9 MB each; they are the design source, not what the shop
sends to a browser. ``tools/build_image_variants.py`` pre-renders two
derivatives which are what the containers actually ship:

    thumb   420px  -- listing grids, cart rows, recommendation widgets
    full   1100px  -- the product detail hero

Keeping the mapping here (rather than rewriting ``image_url`` in the
database) means the stored value stays the stable identity of a product
image, the variants can be re-tuned by re-running the build script, and no
migration or re-seed is needed for databases already in the field.
"""
from __future__ import annotations

_PRODUCTS_PREFIX = "/static/images/products/"


def variant_url(image_url: str, variant: str) -> str:
    """Return the ``variant`` WebP URL for a canonical product image URL.

    Args:
        image_url: The stored product image path.
        variant: The variant directory name, ``"thumb"`` or ``"full"``.

    Returns:
        The URL of the pre-rendered derivative. Anything that is not a
        local product image -- an empty value, or a path that does not sit
        under the products root -- is returned unchanged so an unusual
        product still renders instead of 404ing.
    """
    if not image_url or not image_url.startswith(_PRODUCTS_PREFIX):
        return image_url
    stem = image_url[len(_PRODUCTS_PREFIX):].rsplit(".", 1)[0]
    return f"/static/images/{variant}/{stem}.webp"


def thumb_url(image_url: str) -> str:
    """420px derivative: grids, cart rows, recommendation widgets."""
    return variant_url(image_url, "thumb")


def full_url(image_url: str) -> str:
    """1100px derivative: the product detail hero."""
    return variant_url(image_url, "full")


def register_filters(templates) -> None:
    """Expose ``| thumb`` and ``| full`` to Jinja templates."""
    templates.env.filters["thumb"] = thumb_url
    templates.env.filters["full"] = full_url
