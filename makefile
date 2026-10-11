.PHONY: sync dev lsp lab kernel clean hooks test test-fast golden
sync:
	uv sync --frozen

dev:
	uv sync --group dev

lsp:
	uv sync --group lsp

kernel: dev
	uv run python -m ipykernel install --user --name=pmule --display-name="pmULE"

lab: sync dev lsp
	uv run jupyter lab

clean:
	rm -rf .venv
	uv cache clean

hooks:
	ln -sf ../../scripts/pre-commit-jupytext.sh .git/hooks/pre-commit
	@echo "hook pre-commit instalado (sincroniza pares jupytext de notebooks/)"

# Suite completa: unitarios (src/test_*.py) + regresión de cuadernos (src/tests_ejercicios)
test: dev
	cd src && ../.venv/bin/python -m pytest . -q -p no:cacheprovider

# Solo unitarios (segundos); omite la ejecución de los 24 cuadernos
test-fast: dev
	cd src && ../.venv/bin/python -m pytest . -q -p no:cacheprovider --ignore=tests_ejercicios

# Regenera los golden de los cuadernos (tras un cambio deliberado de comportamiento)
golden: dev
	cd src && ../.venv/bin/python -m tests_ejercicios.make_golden --all
