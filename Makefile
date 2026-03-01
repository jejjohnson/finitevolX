.PHONY: help install uv-format uv-lint uv-pre-commit test
.DEFAULT_GOAL = help

# ANSI Color Codes for pretty terminal output
BLUE   := \033[36m
YELLOW := \033[33m
GREEN  := \033[32m
RED    := \033[31m
RESET  := \033[0m

PYTHON = python
ROOT = ./
SHELL = bash
PKGROOT = finitevolx

help:## Display this help
@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Installation
.PHONY: install
install: ## Install all project dependencies
@printf "$(YELLOW)>>> Initiating environment synchronization and dependency installation...$(RESET)\n"
@uv sync --all-extras
@uv run pre-commit install
@printf "$(GREEN)>>> Environment is ready and pre-commit hooks are active.$(RESET)\n"

##@ Formatting
.PHONY: uv-format
uv-format: ## Run ruff formatter
@printf "$(YELLOW)>>> Formatting code with ruff...$(RESET)\n"
@uv run ruff format $(PKGROOT)
@uv run ruff check --fix $(PKGROOT)
@printf "$(GREEN)>>> Codebase formatted successfully.$(RESET)\n"

##@ Linting
.PHONY: uv-lint
uv-lint: ## Run ruff check and ty
@printf "$(YELLOW)>>> Executing static analysis and type checking...$(RESET)\n"
@uv run ruff check $(PKGROOT)
@uv run ty check $(PKGROOT)
@printf "$(GREEN)>>> Linting checks passed.$(RESET)\n"

##@ Pre-commit
.PHONY: uv-pre-commit
uv-pre-commit: ## Run all pre-commit hooks
@printf "$(YELLOW)>>> Running pre-commit hooks on all files...$(RESET)\n"
@uv run pre-commit run --all-files
@printf "$(GREEN)>>> Pre-commit checks passed.$(RESET)\n"

##@ Testing
.PHONY: test
test: ## Test code using pytest.
@printf "$(YELLOW)>>> Launching test suite with verbosity...$(RESET)\n"
@uv run pytest tests -v
@printf "$(GREEN)>>> All tests passed.$(RESET)\n"
