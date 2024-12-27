MIN_PYTHON_VERSION := "3.9"
MAX_PYTHON_VERSION := "3.13"

_UV_BINARY_VERSION := $(shell uv --version 2>/dev/null)
_RUST_BINARY_VERSION := $(shell cargo --version 2>/dev/null)

.PHONY: setup-environment
setup-environment: ## Install the binaries required to create a python environment
ifndef _UV_BINARY_VERSION
	@echo "UV is not available. Trying to install it."
	@curl -LsSf https://astral.sh/uv/install.sh | sh
endif
ifndef _RUST_BINARY_VERSION
	@echo "Rust is not available. Trying to install it."
	@curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
	@. "$HOME/.cargo/env"
endif
	@echo "UV is configured. Version: $(uv --version)"
	@echo "Rust is configured. Version: $(cargo --version)"


.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@make setup-environment
	@echo "🚀 Creating virtual environment using uv"
	@uv venv -p $(MIN_PYTHON_VERSION)
	@uv pip install -r requirements-dev.txt -e .
	@uv run pre-commit install


.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest
