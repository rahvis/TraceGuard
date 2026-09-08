# Targets for the published artifact. Every target here runs against what this
# repository actually contains. The development repository additionally builds
# the manuscript; those targets are deliberately absent because the LaTeX
# sources are not part of this artifact.

SHELL := /bin/bash
PY ?= uv run
SEED ?= 20260710
HEADLINE ?= artifacts/cvm-honest.jsonl
HIST ?= artifacts/experiment-usenix26-main.jsonl
PERMS ?= 2000

.PHONY: help install test lint doctor offline examples analysis figures \
        matrix family deadline wire citations experiment judge deploy-azure \
        docker-build docker-up docker-down clean

help:
	@echo "Setup and checks"
	@echo "  make install        install the package and dev extras"
	@echo "  make test           run the test suite"
	@echo "  make lint           run ruff"
	@echo "  make doctor         report whether this host is configured offline or live"
	@echo ""
	@echo "Offline, no key and no network"
	@echo "  make offline        deterministic fixture run of the full suite"
	@echo "  make examples       run the four SDK examples"
	@echo ""
	@echo "Reanalysis of the shipped journals, no key and no network"
	@echo "  make analysis       regenerate the attack tables and macros"
	@echo "  make matrix         regenerate the generality sweep tables"
	@echo "  make family         regenerate the second task family tables"
	@echo "  make deadline       regenerate the deadline provisioning tables"
	@echo "  make wire           regenerate the packet level cross check"
	@echo "  make figures        regenerate the figures"
	@echo "  make citations      resolve every arXiv identifier in refs.bib"
	@echo ""
	@echo "Live, requires a configured model deployment and spends money"
	@echo "  make experiment     fresh live replication into artifacts/local"
	@echo "  make judge          judged utility run into artifacts/local"
	@echo ""
	@echo "Deployment"
	@echo "  make deploy-azure   provision an attested confidential VM"

install:
	uv sync --extra dev

test:
	$(PY) pytest

lint:
	$(PY) ruff check .

doctor:
	$(PY) traceguard doctor

offline:
	$(PY) traceguard experiment --provider fixture --n-per-cell 2 --seed $(SEED)
	@echo "Fixture output is a plumbing check labelled synthetic_fixture."
	@echo "It is not evidence that a real agent leaks."

examples:
	@for f in sdk/examples/*.py; do echo "== $$f"; $(PY) python "$$f" || exit 1; done

analysis:
	$(PY) python scripts/make_paper_artifacts.py \
	  --journal $(HEADLINE) --out-dir tables --seed $(SEED) \
	  --nulls tables/nulls.json \
	  --historical-journal $(HIST) --historical-nulls tables/nulls-historical.json

matrix:
	$(PY) python scripts/make_matrix_artifacts.py \
	  --journal-dir artifacts --out-dir tables --seed $(SEED) \
	  --permutations $(PERMS) --report tables/matrix-report.json

family:
	$(PY) python scripts/make_family_artifacts.py \
	  --journal-dir artifacts --out-dir tables --seed $(SEED) --permutations $(PERMS)

deadline:
	$(PY) python scripts/make_deadline_artifacts.py \
	  --journal 4000=artifacts/cvm-dl4000.jsonl \
	  --journal 6000=artifacts/cvm-dl6000.jsonl \
	  --journal 10000=$(HEADLINE) \
	  --headline 10000=$(HEADLINE) --runs artifacts/cvm-runs --tables tables

wire:
	$(PY) python scripts/make_wire_artifacts.py \
	  --journal artifacts/wire/wire.jsonl --runs artifacts/wire/runs \
	  --packets artifacts/wire/wire-packets.txt \
	  --out-dir tables --report artifacts/wire/wire-report.json
	$(PY) python scripts/make_wire_figure.py \
	  --journal artifacts/wire/wire.jsonl --runs artifacts/wire/runs \
	  --packets artifacts/wire/wire-packets.txt \
	  --report artifacts/wire/wire-report.json --out figures/fig_wire.pdf

figures:
	$(PY) python scripts/make_paper_figures.py \
	  --journal $(HEADLINE) --nulls tables/nulls.json --out-dir figures
	$(PY) python scripts/make_asymmetry_figure.py \
	  --journal $(HEADLINE) \
	  --deadline-journal 4000=artifacts/cvm-dl4000.jsonl \
	  --deadline-journal 6000=artifacts/cvm-dl6000.jsonl \
	  --runs artifacts/cvm-runs --out figures/fig_asymmetry.pdf

citations:
	$(PY) python scripts/verify_citations.py --bib refs.bib

experiment:
	@mkdir -p artifacts/local
	$(PY) traceguard experiment --provider azure --n-per-cell 1 --seed $(SEED) \
	  --journal artifacts/local/replication.jsonl

judge:
	@mkdir -p artifacts/local
	$(PY) python scripts/judge_utility.py --n-per-cell 3 --seed $(SEED) --workers 4 \
	  --scores artifacts/local/utility.jsonl --journal $(HEADLINE) --tables-dir tables

deploy-azure:
	cd deploy/azure && ./deploy.sh

docker-build:
	docker compose build

docker-up:
	docker compose up

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ artifacts/local
