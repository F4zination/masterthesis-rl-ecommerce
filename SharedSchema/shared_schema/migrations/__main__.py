"""Module entrypoint for: python -m shared_schema.migrations ..."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
