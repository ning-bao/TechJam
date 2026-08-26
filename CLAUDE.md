# CLAUDE.md — rules for ALL agents working in this repo

Project: Track 5 hackathon — robust AIGC-image detection under transforms.
Spec documents, in order of authority:
1. `PLAN.md` — binding decisions (D1–D12), repo layout, timeline. Do not deviate.
2. `INTERFACES.md` — binding API/schema contracts between modules. Do not deviate.
3. `research/FINAL-track5-merged-research-report.md` — background reference only.

## Discipline (non-negotiable)

- **Do ONLY the task you were given.** Do NOT do things you were not told to do.
  No extra features, no speculative abstractions, no config options nobody asked
  for, no refactors, no renames, no "improvements" to files outside your task,
  no drive-by fixes. If your task seems to require changing a file you don't
  own, STOP and report it instead.
- **File ownership.** Your task message lists the files you own. Touch nothing
  else. Never edit: `CLAUDE.md`, `PLAN.md`, `INTERFACES.md`, `pyproject.toml`,
  another agent's modules.
- **No new dependencies.** The dependency set is pinned in
  `track5/pyproject.toml`. If you believe you need something else, report it;
  do not install it.
- **Do not git init, commit, or push.** The orchestrator handles VCS.
- **Do not overthink.** Plain, minimal, working code beats clever code. No
  docstring essays. Comments only for non-obvious constraints.
- **Use ENGLISH in ALL files.**

## Data safety (hard rules)

- `track5/data/raw/**` is **READ-ONLY**. Downloads are in progress there right
  now. Never write, move, rename, delete, extract into, or "clean up" anything
  under it. Scanning/reading is fine; expect partial files and skip them
  gracefully.
- **Protected set (constraint C2):** COCO **val2017** and the WildFake
  **DALL·E family** must NEVER appear as training, selection, or calibration
  data in any code path, config, or default. They exist only in the denylist
  and (val2017×DALL·E-Advanced) the final protected inference run.
- Anything under `data/`, `runs/`, `cache/` is gitignored bulk — never add
  code that assumes it is versioned.

## Environment facts

- Windows 10, Git Bash shell. Use `pathlib`, forward slashes, explicit
  `encoding="utf-8"` on every file open. No symlinks, no `os.symlink`.
- Venv: `F:/Hackathon/.venv` — run Python as
  `F:/Hackathon/.venv/Scripts/python.exe` (uv-managed, Python 3.13).
- GPU: single RTX 5070 Ti, **16 GB** VRAM, Blackwell (sm_120) — torch must be
  the cu128 build. torch may still be installing when you start; code must
  import without a GPU present.
- **Never launch training or any GPU job unless your task explicitly says to.**
  Smoke tests: CPU only, tiny synthetic inputs, < 60 s.

## Determinism

- Every eval-side transform must be byte-deterministic given
  (input bytes, atom name, seed). Per-item seeds come from
  `track5.utils.seed.item_seed(...)` — never `random.random()` on global state,
  never time-based seeds.
- Training-side stochasticity is allowed only in `transforms/train_sampler.py`
  and dataloader shuffling, always through a seeded generator.

## Verification & reporting

- `py_compile` every file you write. Run your own unit tests with the venv
  python if the needed deps are installed; if they aren't yet, say so.
- Report at the end, exactly: (1) files created/modified, (2) verification
  commands run and their real results, (3) deviations from spec with reason,
  (4) open issues. Never claim untested code works. No marketing language.
