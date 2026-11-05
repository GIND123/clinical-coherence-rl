"""`ccrl analyze`: archived posteriors -> tables, a Markdown summary, figures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ccrl.analysis.estimands import analyze
from ccrl.archive.export import posterior_table
from ccrl.panel import items as I
from ccrl.util import now_iso, read_jsonl
from ccrl.workspace import Workspace

SUMMARY_COLS = ["job", "model_key", "effort", "dataset", "n_cases", "parse_rate", "floor", "ceiling", "NOE",
                "NOE_lo", "NOE_hi", "NE_para", "NE_cue", "NE_dup", "sensitive", "dir_agree_gold", "top1", "brier",
                "ece", "knows_vs_states", "auth_exchange_rate", "entropy"]


def run(ws: Workspace, jobs: list[str] | None = None, n_boot: int = 2000) -> Path:
    post = posterior_table(ws)
    if post.empty:
        raise SystemExit("no panel results in the ledger")
    if jobs:
        post = post[post["job"].isin(jobs)]
    items = I.load_items(ws)
    ci_path = ws.items_dir / "case_index.jsonl"
    case_index = {r["case_id"]: r for r in read_jsonl(ci_path)} if ci_path.exists() else {}
    res = analyze(post, items, case_index, n_boot=n_boot)
    out = ws.reports / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    res["estimands"].to_csv(out / "estimands.csv", index=False)
    res["sequential"].to_csv(out / "sequential.csv", index=False)
    res["parse_by_transform"].to_csv(out / "parse_by_transform.csv", index=False)
    if not res["per_case"].empty:
        res["per_case"].to_parquet(out / "per_case.parquet", index=False)
    E = res["estimands"]
    cols = [c for c in SUMMARY_COLS if c in E.columns]
    lines = [f"# Panel analysis ({'DRY RUN — mock data' if ws.dry_run else 'archived panel'})", "",
             f"Generated {now_iso()} from {len(post)} final responses.", "",
             "NOE/NE: normalised effect (0 = retest noise, 1 = a different patient). "
             "Brackets: case-clustered bootstrap 95% CI.", ""]
    if not E.empty:
        lines.append(E[cols].round(4).to_markdown(index=False) if _has_tabulate() else E[cols].round(4).to_string(index=False))
    if not res["sequential"].empty:
        lines += ["", "## Sequential evidence", "",
                  res["sequential"].round(4).to_string(index=False)]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _figures(E, out)
    return out


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def _figures(E: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if E.empty or "NOE" not in E:
        return
    for ds, g in E[E["job"] == "panel_main"].groupby("dataset") if "panel_main" in set(E["job"]) else E.groupby("dataset"):
        g = g.sort_values("NOE")
        fig, ax = plt.subplots(figsize=(6, 0.35 * len(g) + 1.2))
        y = np.arange(len(g))
        ax.errorbar(g["NOE"], y, xerr=[g["NOE"] - g["NOE_lo"], g["NOE_hi"] - g["NOE"]], fmt="o", color="#1f4e79",
                    ecolor="#7f9cbf", capsize=2)
        ax.axvline(0, color="#999", lw=0.8, ls="--")
        ax.set_yticks(y, [f"{m} ({e})" for m, e in zip(g["model_key"], g["effort"])])
        ax.set_xlabel("normalised order effect (0 = retest noise, 1 = different patient)")
        ax.set_title(f"Order effect — {ds}")
        fig.tight_layout()
        fig.savefig(out / f"fig_noe_{ds}.png", dpi=160)
        plt.close(fig)
