# Current Setup Review

## What looks good
- The repository is logically grouped by workflow area (`meta_arx`, `3DC`, `Tuning`) and includes dedicated setup and simulation readmes.
- A reproducible conda environment file exists (`environment.yml`) with common scientific Python tooling and linters.

## Gaps and risks
1. **Naming/path drift between docs and code**
   - `readme_VRFT_simulation.md` documents a package named `run_simulation_offline_VRFT`, but this repo currently uses `meta_arx/run_simulation_PID`.
   - `meta_arx/run_simulation_PID/scripts/run_vrft_pid.py` still imports from `run_simulation_offline_VRFT`, which currently breaks execution.

2. **Two competing environment setup paths**
   - `readme_package_setup.md` gives manual `conda install ...` steps.
   - `environment.yml` defines a separate, named environment (`plsr-metamodel`) and includes additional developer tools.
   - This can lead to inconsistent local environments.

3. **Execution assumptions are implicit**
   - Script paths in `run_closed_loop.py` are relative to `run_simulation_PID/...`, which only works from specific working directories.
   - There is no top-level run entry point that normalizes cwd assumptions.

4. **No quick validation script**
   - There is no single smoke-test command documented that verifies imports, model files, and script wiring in one step.

## Recommended next actions
- Align naming/imports so docs and code consistently use one package name (prefer current `run_simulation_PID` tree).
- Pick one setup source of truth: either generate `environment.yml` from docs or update docs to `conda env create -f environment.yml`.
- Add a lightweight `scripts/smoke_test.py` (or pytest) that checks critical imports and expected file presence.
- Add one “from clean clone to first run” command sequence at repo root.
