## Agent skills

### Issue tracker

Issues and specs for this repo are tracked using GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

The default set of triage labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) are used in this repo. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository (`GLOSSARY.md` + `docs/adr/` at the repo root). See `docs/agents/domain.md`.

### Runtime layout & core freeze

- Public runtime API headers live under `include/harmony/` (`namespace harmony`) and implementation sources under `src/`, compiled into `lib/libharmony.a`, `bin/`, and `tests/bin/` via `./build.sh all`.
- Layer 2 evaluation and validation harnesses (`job_harmony_super.pbs`, `benchmarks/`, `colab/`) MUST NOT modify any files under `include/harmony/` or `src/` (frozen against baseline `8d7c3392c8d616997b342f778ca041a9a717aff2`).

### Coding standards & review

Review standards and artifact conventions are documented in `CODING_STANDARDS.md`.

### Invariant checks

Run `bash tests/check_invariants.sh` to verify frozen runtime core invariants, C++ and Python unit tests, prebuilt benchmark harness guards, and glossary domain terminology before proposing or merging changes.
