# Runbook

My personal guide to setting up and working in EDF-DNN.

## Where things live

| What | Where |
|---|---|
| Repo | `~/projects/EDF-DNN` (WSL Linux filesystem, fast) |
| Python + all packages | `~/projects/EDF-DNN/.pixi/envs/default/` |
| Package list and versions | `pixi.toml` (what I ask for), `pixi.lock` (what got installed) |
| My own code | `src/edf_dnn/`, importable from anywhere as `edf_dnn` |
| Input data | `data/` (not in git) |

What every folder is for: [repo-description.md](repo-description.md).
ML terms: [glossary.md](glossary.md).

Open the repo from a WSL terminal with `code ~/projects/EDF-DNN`.
Avoid working from `/mnt/c/...`: the Windows drive is about 10x slower from WSL.

## pixi essentials

```bash
pixi install              # build/refresh the environment from pixi.lock
pixi add xarray           # add a package (updates pixi.toml + pixi.lock)
pixi remove xarray        # remove a package
pixi update               # move packages to newer versions within the allowed ranges
pixi list                 # what's installed, with versions
pixi run python script.py # run a script with the project's Python
pixi run test             # run all tests (shortcut defined in pixi.toml)
pixi run ipython          # interactive Python console in the environment
pixi self-update          # update pixi itself
```

Rule of thumb: always go through `pixi run ...`. Plain `python` in a terminal
is Ubuntu's own Python, which has none of the packages.

On a GPU machine (e.g. NCAR Casper): `pixi install -e cuda`, then `pixi run -e cuda ...`.

## One-time VS Code setup

1. Ctrl+Shift+P → **Python: Select Interpreter** → choose
   `~/projects/EDF-DNN/.pixi/envs/default/bin/python`.
2. Ctrl+Shift+P → **Preferences: Open Workspace Settings (JSON)** (this
   creates `.vscode/settings.json` in the repo) and add:

   ```json
   "python.terminal.launchArgs": [
       "-m", "IPython", "--no-autoindent",
       "--ext=autoreload", "--InteractiveShellApp.exec_lines=%autoreload 2"
   ]
   ```

   This makes Shift+Enter send code to an IPython console with autoreload
   already switched on, so I never need to type `%load_ext autoreload` /
   `%autoreload 2` myself.
   Workspace, not User settings: User settings also apply to every other
   Python project (including Windows ones), which may not have IPython.
   `.vscode/` is gitignored, so redo this step after a fresh clone.
3. After a Python version change: `pixi run register-kernel` (for Quarto).

## Working interactively, R-style

- **Shift+Enter** runs the current line or selection, like Ctrl+Enter in RStudio.
- **Autoreload** (set up above) means edits to `src/edf_dnn/` apply on the next
  call without restarting the console, like `devtools::load_all()`.
  If the console was started some other way, type once:
  `%load_ext autoreload` then `%autoreload 2`.
- The console starts in the repo root, so paths like `data/...` work.
- **Stepping through a function**: copy its body into a file in
  `manual_zone/scratch/`, define its arguments by hand at the top, and run it
  line by line. Example: `manual_zone/scratch/verify_grid.py`.
- **Pausing inside a function**, like R's `browser()`: put `breakpoint()` on a
  line. Type variable names to inspect them, `n` for next line, `c` to continue,
  `q` to quit.
- **After an error**, type `%debug` in IPython to jump to the failing line with
  all its variables, like R's `traceback()` + `recover()`.
- Restart the console when things get weird: click the bin icon on the
  terminal, then Shift+Enter starts a fresh one.

## Checking the pipeline step by step

Scripts in `manual_zone/scratch/`, in pipeline order, to step through with
Shift+Enter on the real data:

1. `verify_grid.py`: load the rasters, crop to the future grid, pixel table
2. `verify_splits.py`: test set, folds and downsampling, with maps
3. `verify_normalise.py`: scaling (each variable on its own 0-1 scale)
4. `reprex_batches.py`: mini-batches, on ten made-up pixels
5. `walkthrough_mlp.py`: the whole MLP for one fold, top to bottom (about a minute)
6. `verify_training.py`: training numbers against known answers (loss at start, memorising 64 pixels, logistic-regression baseline, early stopping; about 3 minutes)
7. `verify_run.py`: a finished run folder's files against each other and the data (`runs/dev_quick/` by default; about 15 seconds)

The tests for one stage: `pixi run pytest tests/test_splits.py -v`.

## R to Python cheat sheet

| R | Python |
|---|---|
| `str(x)` / print | `x` (just type the name) |
| `dim(x)` | `x.shape` |
| `names(r)` (raster) | `stack.band.values` |
| `r[["tcc"]]` | `stack.sel(band="tcc")` |
| `plot(r[["tcc"]])` | `stack.sel(band="tcc").plot()` |
| `head(df)` | `df.head()` |
| `summary(df)` | `df.describe()` |
| `class(x)` | `type(x)` |
| `?fun` | `fun?` (IPython) |
| `ls()` | `%whos` (IPython) |
| `x[1]` | `x[0]` (Python counts from 0) |

## Known harmless messages

- `Error in sys.excepthook: Original exception was:` with nothing after it,
  when a script using rioxarray exits. Data are fine. See NOTES.md.
