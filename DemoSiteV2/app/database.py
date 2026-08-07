from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import DATABASE_URL
from shared_schema.migrations import run_migrations

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def _sqlite_path_from_url(database_url: str) -> str:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError(f"Only sqlite:/// URLs are supported for migrations: {database_url}")
    return database_url[len(prefix):]


def get_db():
    """Yield a SQLAlchemy database session and close it after use.

    Intended as a FastAPI dependency injected via ``Depends(get_db)``.
    Yields a ``Session`` object for the duration of a single request,
    then closes it in the ``finally`` block regardless of success or error.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all database tables defined in the ORM models.

    Imports the ``models`` module to ensure all SQLAlchemy mappers are
    registered before calling ``Base.metadata.create_all``.  Safe to call
    on every startup — SQLAlchemy skips tables that already exist.
    """
    from . import models  # noqa: F401 – ensure models are registered
    Base.metadata.create_all(bind=engine)
    db_path = _sqlite_path_from_url(DATABASE_URL)
    run_migrations(db_path)
