# Project-I: Energy Data Pattern Detection & AutoML

## Project Overview
University study project focused on discovering consumption regimes in energy data from a university building (Prague, Czech Republic). Data is recorded at 15-minute intervals and includes energy consumption, outside temperature, and timestamps.

**Goal**: Identify regimes (semester periods, holidays, exam periods, summer, COVID, etc.) from the data, then apply AutoML to select and train the most accurate model per regime — ultimately for use on online (live) data.

**Current phase**: Exploratory — understanding whether regime-based modeling is feasible. Target variable is not yet defined.

## Data Context
Claude cannot access `data/` but here is the schema for reference:

- **Index**: `timestamp` — DatetimeIndex with timezone (CET/CEST), 15-min intervals, 2016-03-09 to 2026-03-09 (~350k rows)
- **Columns**:
  | Column | Dtype | Non-Null | Notes |
  |---|---|---|---|
  | `fve_d1_power_kw` | float64 | 347k | PV system D1 power output |
  | `fve_vvn_power_kw` | float64 | 346k | PV system VVN power output |
  | `main_meter_power_kw` | float64 | 343k | Main meter — has extreme outliers (max ~2.3e9, likely errors) |
  | `solar_irradiance_wm2` | float64 | 346k | Solar irradiance; some negative values |
  | `motors_power_kw` | float64 | 348k | Motors subsystem |
  | `lights_power_kw` | float64 | 231k | Lights subsystem — significant missing data (~34%) |
  | `virtual_solar_irradiance` | float64 | 0 | Entirely empty — ignore |
  | `daytime` | str | 350k | Categorical (e.g. "AAAA") — meaning TBD |
  | `temp_c` | float64 | 88k | Outside temperature — ~75% missing, range -16 to 37 C |

- **Key data quality notes**:
  - `main_meter_power_kw` has impossible values (max 2.3 billion kW) — needs outlier handling
  - `temp_c` is very sparse (~25% coverage)
  - `lights_power_kw` missing ~1/3 of values
  - `virtual_solar_irradiance` is entirely null — can be dropped
  - Some PV columns have small negative values (sensor noise)

## Tech Stack
- **Python 3.12** with **Poetry** for dependency management
- Standard data science libraries (pandas, numpy, matplotlib, scikit-learn, etc.)
- AutoML framework: TBD (to be explored during the project)
- Add all new dependencies via `poetry add`

## Project Structure
```
Project_I/
├── CLAUDE.md
├── pyproject.toml
├── data/              # Raw and processed data — OFF LIMITS (see rules)
├── notebooks/         # Final Jupyter notebooks (.ipynb) for presentation
├── experiments/       # Experimental .py working scripts
├── src/project_i/     # Reusable functions and modules
└── tests/
```

## Workflow Rules
1. **Always enter plan mode** before starting any non-trivial task
2. **Never access `data/`** — do not read, modify, or list files in the data directory
3. **Do not assume data structure** — ask the user about columns, types, and formats
4. **Explain non-obvious decisions** — the user has ML/DL/data science experience, so skip basics but explain architectural or algorithmic choices that aren't standard
5. **English only** for code, comments, variable names, and commit messages
6. **Solo project** — simple git workflow on main branch

## Code Style
- **Experimental scripts**: Plain `.py` files in `experiments/`. No `# %%` cell markers — the user selects and runs lines via Shift+Enter in VS Code interactive window
- **Presentation notebooks**: `.ipynb` files in `notebooks/`, created when results need to be presented
- **Reusable code**: Functions and modules go in `src/project_i/`
- Keep code style basic and standard for now — the style guide will evolve over time
- Keep it simple — no over-engineering, no unnecessary abstractions

## What Claude Must NOT Do
- Read or modify anything in `data/`
- Make assumptions about data structure without being told
- Over-engineer solutions or add unnecessary abstractions
- Add features, refactoring, or "improvements" beyond what was asked
