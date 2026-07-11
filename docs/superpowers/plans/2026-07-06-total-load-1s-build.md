# Total Load 1s Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a traceable 1 s total-load Excel dataset from the existing 30 s energy-side `total_load_excels` plus linearly interpolated AIS speed.

**Architecture:** Add one standalone builder script that reads existing 30 s total-load Excel files, resamples numeric energy-side columns to a 1 s timestamp grid, linearly interpolates AIS speed from the matching `氢舟一号/<voyage>/推进系统/AIS*.csv`, writes one Excel workbook per voyage under `total_load_excels_1s`, and writes a summary CSV. Do not touch LSTM/MPC training code.

**Tech Stack:** Python, pandas, existing standard-library XLSX reader pattern, a small standard-library XLSX writer, unittest.

---

### Task 1: Builder Tests

**Files:**
- Create: `tests/test_build_total_load_1s_excels.py`

- [x] **Step 1: Write failing tests**

Tests cover:
- 1 s grid generation from 30 s input.
- Linear interpolation for `fuel_cell_total_kw`, `battery_total_kw`, `total_load_fc_plus_batt_kw`, and `speed_knots`.
- `total_load_fc_plus_batt_kw = fuel_cell_total_kw + battery_total_kw` after interpolation.
- Missing or mismatched voyage names fail loudly.

### Task 2: Builder Implementation

**Files:**
- Create: `src/main/build_total_load_1s_excels.py`

- [x] **Step 1: Implement minimal builder**

Required public functions:
- `build_one_voyage_1s(...)`
- `build_total_load_1s_excels(...)`
- `write_xlsx(...)`

Required CLI:
```powershell
D:\py\Python3\python.exe src\main\build_total_load_1s_excels.py --input_dir total_load_excels --ais_root C:\Users\20883\OneDrive\Desktop\氢舟一号 --output_dir total_load_excels_1s --summary_dir outputs\total_load_1s_build
```

### Task 3: Generate Dataset

**Files:**
- Output: `total_load_excels_1s/*.xlsx`
- Output: `outputs/total_load_1s_build/summary_total_load_1s.csv`

- [x] **Step 1: Run builder**
- [x] **Step 2: Verify 66 output files and summary checks**

### Task 4: Status Updates

**Files:**
- Modify: `project_status.md`
- Modify: `next_steps.md`
- Modify: `thread.md`

- [x] **Step 1: Record output paths and validation results**
