# Migrating to Vestige 4.0

4.0 boots a Strata log. It detects a v3 store only when `vestige.db` exists. It does not open that file.

When `vestige-upgrade` is next to the `vestige` binary or on `PATH`, 4.0 runs it. `vestige-upgrade` holds the store lock, imports into a staging log, verifies that log, and renames it into place. `vestige.db` is left byte-identical.

When `vestige-upgrade` is missing, 4.0 refuses to start and leaves `vestige.db` untouched.

## Rollback

Rollback is reinstalling v3.1.1 from GitHub releases:

<https://github.com/samvallad33/vestige/releases/tag/v3.1.1>

The v3 `vestige.db` is left untouched by the upgrade. Point that v3.1.1 install at the same data directory and it opens the original file.
