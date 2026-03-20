.PHONY: install dev-install up down ingest eval ui lint typecheck test clean

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

# ── Core workflows ─────────────────────────────────────────────────────
ingest:
	@test -n "$(PDF_DIR)" || (echo "Usage: make ingest PDF_DIR=./data/papers" && exit 1)
	python -m papermind.scripts.ingest --pdf-dir $(PDF_DIR)

eval:
	python -m papermind.scripts.evaluate --test-set data/test_set/questions.json

ui:
	streamlit run src/papermind/ui/app.py

# ── Quality ────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/

test:
	pytest

test-verbose:
	pytest -v --tb=short

# ── Cleanup ────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build *.egg-info
