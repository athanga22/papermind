.PHONY: install dev-install up down ingest query lint test clean

# ── Setup ──────────────────────────────────────────────────────────────
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

# ── Infrastructure ─────────────────────────────────────────────────────
up:
	docker compose up -d
	@echo "Qdrant ready at http://localhost:6333"

down:
	docker compose down

# ── Step 1 workflows ───────────────────────────────────────────────────
ingest:
	@test -n "$(PDF_DIR)" || (echo "Usage: make ingest PDF_DIR=./data/papers" && exit 1)
	python -m papermind.scripts.ingest --pdf-dir $(PDF_DIR)

ingest-fresh:
	@test -n "$(PDF_DIR)" || (echo "Usage: make ingest-fresh PDF_DIR=./data/papers" && exit 1)
	python -m papermind.scripts.ingest --pdf-dir $(PDF_DIR) --recreate

query:
	@test -n "$(Q)" || (echo 'Usage: make query Q="your question here"' && exit 1)
	python -m papermind.scripts.query "$(Q)"

# ── Quality ────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

test:
	pytest

# ── Cleanup ────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
