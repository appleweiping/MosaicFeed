# Contributing

Thanks for improving MosaicFeed. Open an issue before a large behavioral change so the point-in-time and explanation contracts can be discussed first.

1. Create a focused branch.
2. Add tests that fail without the change.
3. Run `ruff check .`, `mypy src`, the coverage command from the README, and `python -m build`.
4. Update user-facing documentation when behavior or JSON changes.
5. Keep runtime dependencies at zero unless an issue documents a compelling reason.

Bug reports should include a minimal catalog, event log, configuration, command, actual output, and expected output. Never include real private interaction logs.
