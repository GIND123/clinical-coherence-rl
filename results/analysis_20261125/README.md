# Analysis tables, 25 Nov 2026

Produced by `ccrl analyze` (CCRL_RUNS pointing at the ledger in the PI's archive
`ccrl-archive-20261125-144504.tar.gz`, SHA-256 `18782203c3ddc24fd75a058adb93ee650c8b541b15c339e42dab997525d4624e`),
except `identical_text.csv`: the share of byte-identical response texts to byte-identical
requests (base vs retest), on the core cases, computed from the ledger's `attempts.text`.

| file | contents |
|---|---|
| `estimands.csv` | per job x model x effort x dataset: NOE and normalised effects, accuracy, calibration, authority |
| `stability.csv` | retest flip, ensemble gain, variance share, AUROC confidence vs disagreement |
| `updating.csv` | `resample`: evidence updating on mean posteriors |
| `paired_vs_panel.csv` | `temp0` and `effort_arm` vs the main panel on the same cases |
| `sequential.csv`, `sequential_gap.csv` | two-stage evidence: order gap and its CI beyond the retest floor |
| `parse_by_transform.csv` | parse rates |

Interpretation: docs/results_panel_main.md and docs/results_secondary.md.
