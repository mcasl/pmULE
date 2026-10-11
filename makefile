.PHONY: sync dev lsp lab kernel clean hooks
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
