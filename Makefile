.PHONY: install run debug clean lint

export HF_HOME=$(HOME)/sgoinfre

export UV_CACHE_DIR=$(HOME)/sgoinfre/uv_cache

export UV_PROJECT_ENVIRONMENT := $(HOME)/sgoinfre/venv

unexport VIRTUAL_ENV

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache src/__pycache__ src/llm_sdk/__pycache__ data/output/*

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

moulinette_private:
	uv run python -m moulinette prepare_exercises --output="data" --set="private"

moulinette_grade_private:
	uv run python -m moulinette grade_student_answers data/output/function_calling_results.json --set="private"
