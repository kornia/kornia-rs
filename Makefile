.DEFAULT_GOAL := help

PYTHONPATH=
SHELL=/bin/bash
VENV=.venv

ifeq ($(OS),Windows_NT)
	VENV_BIN=$(VENV)/Scripts
else
	VENV_BIN=$(VENV)/bin
endif

# Define command to filter pip warnings when running maturin
FILTER_PIP_WARNINGS=| grep -v "don't match your environment"; test $${PIPESTATUS[0]} -eq 0

.venv:  ## Set up Python virtual environment and install requirements
	python3 -m venv $(VENV)
	$(MAKE) requirements

.PHONY: requirements
requirements: .venv  ## Install/refresh Python project requirements
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/python -m pip install -r py-kornia/requirements-dev.txt

.PHONY: build-python
build-python: .venv  ## Compile and install Python Polars for development
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate \
	&& maturin develop -m py-kornia/Cargo.toml \
	$(FILTER_PIP_WARNINGS)

.PHONY: build-python-release
build-python-release: .venv  ## Compile and install a faster Python Polars binary with full optimizations
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate \
	&& maturin develop -m py-kornia/Cargo.toml --release \
	$(FILTER_PIP_WARNINGS)

.PHONY: test-python
test-python: .venv  ## Run Python tests
	@unset CONDA_PREFIX && source $(VENV_BIN)/activate \
	&& maturin develop -m py-kornia/Cargo.toml \
	&& $(VENV_BIN)/pytest py-kornia/tests
	$(FILTER_PIP_WARNINGS)

.PHONY: clippy
clippy:  ## Run clippy with all features
	cargo clippy --workspace --all-targets --all-features --locked -- -D warnings

.PHONY: clippy-default
clippy-default:  ## Run clippy with default features
	cargo clippy --all-targets --locked -- -D warnings

.PHONY: fmt
fmt:  ## Run autoformatting and linting
	cargo fmt --all

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@rm -rf .venv/
	@rm -rf target/
	@rm -f Cargo.lock
	@cargo clean
	@$(MAKE) -s -C py-kornia/ $@

.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' | sort
