"""The CLI must import and parse every command (catches syntax errors in
modules the other tests do not import), and every module must compile."""
import compileall
from pathlib import Path

import pytest

from ccrl import cli

SRC = Path(__file__).resolve().parents[1] / "src" / "ccrl"


def test_every_module_compiles():
    assert compileall.compile_dir(str(SRC), quiet=1, force=True)


@pytest.mark.parametrize("argv", [
    ["doctor", "--offline"], ["models", "probe", "--models", "opus-5.5"], ["prep", "mcr", "--roles", "pilot"],
    ["build", "items"], ["freeze", "--check"], ["plan"], ["run", "pilot", "--limit-cases", "2"], ["status"],
    ["report", "pilot"], ["export"], ["archive", "pack"], ["billing", "reconcile", "--start", "2026-11-01"],
    ["analyze"], ["rehearse"], ["train", "run", "--config", "config/train/ccr_acc_fid.yaml"],
    ["train", "reward-suite"], ["--dry-run", "status"], ["run", "partner_panel"],
    ["train", "compare", "--ref", "a.parquet", "--new", "b.parquet", "--ref-model", "qwen3-32b"],
])
def test_parser_accepts(argv):
    args = cli.build_parser().parse_args(argv)
    assert callable(args.fn)


def test_help_runs(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    assert "rehearse" in capsys.readouterr().out


def test_status_dry_run(tmp_runs, capsys):
    assert cli.main(["--dry-run", "status"]) == 0
    assert "Committed" in capsys.readouterr().out
