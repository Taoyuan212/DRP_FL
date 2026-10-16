# Data cleaning policy V1

Official source: [Zhou et al., Scientific Data 11, 649 (2024), DOI:10.1038/s41597-024-03427-5](https://www.nature.com/articles/s41597-024-03427-5). Project rules below are explicitly separated from official rules.

## Main task policy

| Class | Input history | Target label | Delete raw row | Interpolate | Indicator | Robustness branch |
|---|---|---|---|---|---|---|
| Missing Patv/Wspd | Invalidates the complete V1 window | Mask | Never | Main: none | Flags retained in index | Causal fill up to 1 or 2 steps is count-only sensitivity |
| Official unknown power | Keep observed historical Patv/Wspd; it is causally known operating history | Mask from Task B | Never | Never | `history_unknown_count`, `target_official_unknown` | Task A later, not in Stage 2 data |
| Official abnormal record (Wdir/Ndir rule) | Core Patv/Wspd are retained only for history because excluded fields are not V1 inputs | Mask | Never | Never | `target_official_abnormal` | Retain flags for audit |
| Project physical/sensor flag | If it affects Wspd, invalidates V1 history; Etmp/Itmp are excluded from V1 | Mask only when the Core Wspd is flagged; an unused temperature sensor alone does not invalidate Patv | Never | Never | Flags retained | Threshold sensitivity required before adding those fields |
| Negative Patv not official-unknown | Historical/target value is clipped to zero in the processed view, raw is unchanged | Valid after clipping | Never | Not applicable | Raw and clipped value retained | Unclipped-score sensitivity later |

Project physical thresholds are audit rules, not official SDWPF definitions: Wspd outside [0.0, 40.0] m/s, and Etmp/Itmp outside [-60.0, 80.0] °C. Temperature fields are excluded from V1, so their flags neither remove a Core-feature history window nor invalidate an otherwise valid Patv target. This avoids manufacturing target selection from an unused sensor; the flags are retained for audit.

## Boundary rules

A sample is invalid if history or direct target crosses a global gap, a sampling-regime boundary, or a split boundary; if history lacks any Core value; or if the target is missing/unknown/abnormal. V1 uses no interpolation. The reusable boolean interface is `is_valid_window(...)` in `scripts/stage2_windows.py`.
