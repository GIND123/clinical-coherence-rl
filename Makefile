# Convenience targets. Every target is a thin wrapper over the `ccrl` CLI.
.PHONY: setup test doctor rehearse plan status reward-suite

setup:
	bash scripts/setup.sh

test:
	.venv/bin/python -m pytest -q

doctor:
	.venv/bin/ccrl doctor

rehearse:
	.venv/bin/ccrl rehearse

plan:
	.venv/bin/ccrl plan

status:
	.venv/bin/ccrl status

reward-suite:
	.venv/bin/ccrl train reward-suite
