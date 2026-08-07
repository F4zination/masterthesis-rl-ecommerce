# SharedSchema

Shared constants, feature normalization utilities, and versioned database migrations for the DemoSite applications.

## Install

```bash
cd SharedSchema
pip install -e .
```

## Database Migrations

Migrations are centralized in `shared_schema.migrations` and can be run from any project folder.

```bash
# Show current version and pending steps
python -m shared_schema.migrations status --db-path ../DemoSiteV2/demosite_test.db

# Apply all pending migrations to latest known version
python -m shared_schema.migrations run --db-path ../DemoSiteV2/demosite_test.db

# Apply migrations up to a specific version
python -m shared_schema.migrations run --db-path ../DemoSiteV2/demosite_test.db --target-version 2
```

Optional console-script entrypoint:

```bash
shared-schema-migrate status --db-path ../DemoSiteV2/demosite_test.db
shared-schema-migrate run --db-path ../DemoSiteV2/demosite_test.db
```

## Runtime Integration

DemoSiteV2 and DemoSiteV3 call `run_migrations(...)` during `init_db()`, so migration checks run automatically on startup after table creation.

## Adding a New Schema Version

1. Create a new module in `shared_schema/migrations/versions/`.
2. Implement an `apply(conn)` function.
3. Register a new `MigrationStep` in `shared_schema/migrations/registry.py`.
4. Bump `CONTEXT_SCHEMA_VERSION` in `shared_schema/constants.py` if context schema changed.
5. Run migration status and migration run commands against test databases.

## Current Migration Steps

- v1 -> v2: add `bandit_arm_stats.schema_version` and tag legacy flat context keys as schema_version=1.
