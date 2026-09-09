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

.PHONY: help install test lint doctor offline examples reproduce analysis \
        figures matrix family deadline wire utility citations experiment judge \
        deploy-azure docker-build docker-up docker-down clean

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
	@echo "  make reproduce      the whole pipeline, then check it against the release"
	@echo "  make analysis       regenerate the attack tables and macros"
	@echo "  make matrix         regenerate the generality sweep tables"
	@echo "  make family         regenerate the second task family tables"
	@echo "  make deadline       regenerate the deadline provisioning tables"
	@echo "  make wire           regenerate the packet level cross check"
	@echo "  make figures        regenerate the figures"
	@echo "  make utility        regenerate the judged utility table offline"
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

# TRACEGUARD_ARTIFACT_DIR is set so the fixture run cannot write into
# artifacts/runs, which holds the signed receipt stores the paper's fail-closed
# rates are recovered from. The default store root is artifacts/runs, so without
# this a documented smoke test would deposit synthetic receipts in the evidence.
offline:
	@mkdir -p artifacts/local/runs
	TRACEGUARD_ARTIFACT_DIR=artifacts/local/runs \
	  $(PY) traceguard experiment --provider fixture --n-per-cell 2 --seed $(SEED)
	@echo "Fixture output is a plumbing check labelled synthetic_fixture."
	@echo "It is not evidence that a real agent leaks."

examples:
	@mkdir -p artifacts/local/runs
	@for f in sdk/examples/*.py; do echo "== $$f"; \
	  TRACEGUARD_ARTIFACT_DIR=artifacts/local/runs $(PY) python "$$f" || exit 1; done

# The entry point the README names. Runs every stage in the order the pipeline
# requires and finishes by checking the regenerated output against the values
# this release committed, so a successful run is a verified reproduction rather
# than merely a completed one.
reproduce:
	bash scripts/reproduce_all.sh

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

# Offline reanalysis of the committed judged run. --analyze-only runs no cells
# and needs no credentials; a cell absent from the scores file is reported as a
# shortfall rather than filled by a stub.
utility:
	$(PY) python scripts/judge_utility.py --analyze-only --n-per-cell 3 \
	  --scores artifacts/utility-cvm-enclave.jsonl --journal $(HEADLINE) \
	  --tables-dir tables

citations:
	$(PY) python scripts/verify_citations.py --bib refs.bib

experiment:
	@mkdir -p artifacts/local
	$(PY) traceguard experiment --provider azure --n-per-cell 1 --seed $(SEED) \
	  --journal artifacts/local/replication.jsonl

# A live judged run writes its tables to tables/local, not to tables. Pointing a
# fresh run at tables would regenerate tab_utility.tex and merge its macros over
# macros.tex, silently replacing the paper's judged values with those of a
# different, smaller run. Reanalysis of the committed run is 'make utility'.
judge:
	@mkdir -p artifacts/local tables/local
	$(PY) python scripts/judge_utility.py --n-per-cell 3 --seed $(SEED) --workers 4 \
	  --scores artifacts/local/utility.jsonl --journal $(HEADLINE) \
	  --tables-dir tables/local

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
