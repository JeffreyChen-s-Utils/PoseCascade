# Release pipeline

Every push to `main` ships a new PyPI release. The pipeline is three
GitHub Actions workflows wired together; this document explains how
they interact, how to opt out of a release for a noisy commit, and
the one-time PyPI configuration the maintainer needs to do.

## Flow at a glance

```
PR opened ──► tests.yml ───────────────► ✅ ruff + bandit + pytest
                                           │
PR merged to main ──► tests.yml ─────────► ✅ run again on the merge commit
                  └──► release.yml ──────► tag v{auto-bumped version}
                                           │
                                           ▼
                       wheels.yml ──────► build OS × Python matrix
                                  ──────► publish to PyPI via Trusted Publishing
```

* **`tests.yml`** — runs `ruff check`, `bandit`, and `pytest` (under
  `xvfb-run` so Qt + offscreen GL fixtures work) on every PR and
  every push to `main`. Three Python versions (3.12 / 3.13 / 3.14)
  on Ubuntu.
* **`release.yml`** — only fires on push to `main`. Reads the most
  recent `v*` tag, bumps the patch component, and pushes a new
  `v{MAJOR.MINOR.PATCH}` tag.
* **`wheels.yml`** — triggers on tag pushes (and PRs that touch the
  build config). Builds wheels through `cibuildwheel` for Win /
  macOS / Linux × cp312 / cp313 / cp314, builds an sdist, then
  uploads to PyPI through Trusted Publishing.

## Bump rules

| Head commit message contains | Bump |
|------------------------------|------|
| `[release major]`            | major (e.g. `v1.2.3` → `v2.0.0`) |
| `[release minor]`            | minor (e.g. `v1.2.3` → `v1.3.0`) |
| (anything else)              | patch (e.g. `v1.2.3` → `v1.2.4`) |
| `[skip release]`             | no release |
| `[skip ci]`                  | no release (and no CI overall) |

The first three are case-insensitive and match anywhere in the
commit subject. The two skip markers exit `release.yml` cleanly
without creating a tag — useful for docs-only commits where shipping
a wheel adds no value.

## Versioning

Versions are derived from git tags via [setuptools-scm](https://github.com/pypa/setuptools_scm).

* On a tag-build (CI on a `v*` tag): version = the tag minus the
  `v` prefix.
* On a non-tag build (local `pip install -e .` from a clean working
  tree): version = `{next-tag}.devN+g{shorthash}` — N is the
  commit-count since the last tag.
* On a non-tag build from an sdist with no git metadata: falls back
  to `0.0.0` (see `[tool.setuptools_scm] fallback_version` in
  `pyproject.toml`).

`local_scheme = "no-local-version"` is set so non-tag builds also
produce PEP 440-clean versions — PyPI / TestPyPI reject the
`+gSHA` local segment setuptools-scm would emit by default.

## One-time PyPI configuration

The publish step uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/),
which means **no API token is stored in the repo**. The maintainer
configures this once on the PyPI side:

1. Sign in to <https://pypi.org/> as a maintainer of the
   `posecascade` project. (If the project doesn't exist yet, the
   first release uses the [pending publisher flow](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
   — PyPI accepts the upload and registers the project on first
   successful publish.)
2. Project → **Manage** → **Publishing**.
3. Add a new GitHub publisher with:
   * Owner: `JeffreyChen-s-Utils`
   * Repository: `PoseCascade`
   * Workflow filename: `wheels.yml`
   * Environment name: `release`
4. Save. The next tag-push that reaches the `publish` job will be
   accepted.

The matching `environment: release` in `wheels.yml` is what PyPI's
side checks against. If you change the environment name in either
file, change it in both.

## Opting out of a release

Three options, in order of preference:

1. **Docs-only commits** — include `[skip release]` in the merge
   commit message. CI still runs (so docs lint stays gated) but no
   wheel is published.
2. **Squash-merge titles** — when GitHub squashes a PR on merge,
   the squash commit's first line becomes the head commit message.
   Edit the squash title to add `[skip release]` before clicking
   merge.
3. **Skip all CI** — `[skip ci]` in the head commit message skips
   every workflow on this commit. Use only when you're certain
   nothing in the change touches code (rare).

## Reverting / yanking a release

PyPI's "Yank" is one-click from the project's release page and is
preferable to deleting the version (PyPI does not allow re-uploading
a previously-published version number). After yanking, push another
commit to `main` and the patch bump produces a new version that
supersedes the yanked one.

If the bad version was published seconds ago and never installed by
anyone, you can delete it from the PyPI project page within a short
window — but yanking is the standard practice and what `pip` honours
correctly.

## Local "what would CI run?" reproduction

```bash
# Same three gates the tests.yml workflow runs.
.venv/Scripts/python.exe -m ruff check .
.venv/Scripts/python.exe -m bandit -c pyproject.toml -r posecascade/
.venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/test_export_video.py
```

To dry-run the version bump:

```bash
# Read the latest tag.
git tag --list 'v*' --sort=-v:refname | head -n1

# Inspect what setuptools-scm would call this build.
.venv/Scripts/python.exe -m setuptools_scm
```

The `setuptools_scm` invocation prints the version string the build
backend would stamp onto the wheel — useful as a sanity check after
landing a tag manually.
