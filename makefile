.PHONY: sync dev lsp lab clean
sync:
	uv sync --frozen

dev:
	uv sync --group dev

lsp:
	uv sync --group lsp

lab: sync dev lsp
	uv run jupyter lab

clean:
	rm -rf .venv
	uv cache clean

