# Releasing pyrosm

Releases are automated by the `release` GitHub Actions workflow
([.github/workflows/release.yaml](.github/workflows/release.yaml)). Pushing a
version tag (`vX.Y.Z`) builds the binary wheels and the sdist, publishes them to
PyPI, and creates the GitHub release.

## Cutting a release

1. Bump `version` in [setup.py](setup.py) (and `version = release` in
   [docs/conf.py](docs/conf.py)) to the new `X.Y.Z`.
2. Move the `Unreleased` notes under a new `vX.Y.Z` heading in
   [CHANGELOG.md](CHANGELOG.md) and [docs/changelog.rst](docs/changelog.rst).
3. Merge the above to `master`.
4. Tag and push — the tag must be `v` + the exact `setup.py` version:

   ```bash
   git checkout master && git pull
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

The workflow then runs: `tests` → `check-version` (the tag must equal the
`setup.py` version, otherwise it fails before building) → build wheels
(Linux/macOS/Windows × CPython 3.10–3.14) + sdist → publish to PyPI → create the
GitHub release. Pre-release tags (e.g. `v0.9.0rc1`) publish and are marked as a
GitHub pre-release automatically.

## Dry run (no publishing)

To verify the wheels build on every platform without releasing: **Actions →
release → Run workflow**. The manual dispatch builds the wheels + sdist but skips
the publish and GitHub-release jobs (those are gated on a tag).

## Notes

- The published version comes from `setup.py`; `check-version` enforces that the
  tag matches it, so a mislabeled version cannot be published.
- `pypa/gh-action-pypi-publish` runs with `skip-existing: true`, so re-running a
  tag whose version is already on PyPI does not error.
