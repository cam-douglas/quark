# Quark Project Makefile
# ======================
# Central build and validation orchestration

.PHONY: help todo plan work track review standup sprint simulate brain-status brain-list brain-analyze brain-test brain-profile brain-visualize brain-orchestrate brain-startup train training-status deploy deployment-status docs update-readme benchmark compare-metrics tasks generate-tasks sync-tasks validate validate-sprint validate-quick validate-ci validate-metrics validate-dashboard validate-rubrics validate-rules validate-sync validate-help clean clean-validation test lint install dev-setup

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║               🚀 QUARK PROJECT - MAKE COMMANDS                   ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "🎯 UNIFIED TODO SYSTEM (Natural Language Interface)"
	@echo "  make todo                  - Show what's next"
	@echo "  make plan                  - Plan new task"
	@echo "  make work [task]           - Work on a task"
	@echo "  make track                 - Track progress"
	@echo "  make review                - Review completed work"
	@echo "  make standup               - Daily standup workflow"
	@echo "  make sprint                - Sprint review workflow"
	@echo ""
	@echo "🧠 BRAIN SIMULATION COMMANDS"
	@echo "  make simulate [component]  - Run brain simulation (cerebellum, cortex, etc)"
	@echo "  make brain-status          - Check brain system status"
	@echo "  make brain-list            - List all brain components"
	@echo "  make brain-analyze [comp]  - Analyze brain component"
	@echo "  make brain-test [comp]     - Test brain component"
	@echo "  make brain-profile         - Profile brain performance"
	@echo "  make brain-visualize       - Visualize brain architecture"
	@echo "  make brain-orchestrate     - Check orchestrator status"
	@echo "  make brain-startup [mode]  - Start brain systems (full/minimal/cognitive_only)"
	@echo ""
	@echo "🚀 TRAINING & DEPLOYMENT"
	@echo "  make train                 - Start model training"
	@echo "  make training-status       - Check training status"
	@echo "  make deploy                - Deploy to platform (local/gcp/docker)"
	@echo "  make deployment-status     - Check deployment status"
	@echo ""
	@echo "📊 BENCHMARKING & DOCS"
	@echo "  make benchmark             - Run performance benchmarks"
	@echo "  make compare-metrics       - Compare benchmark results"
	@echo "  make docs                  - Generate documentation"
	@echo "  make update-readme         - Update README files"
	@echo ""
	@echo "📝 VALIDATION COMMANDS (Manual validation - shows requirements)"
	@echo "  make validate [domain]     - Interactive validation guide"
	@echo "  make validate-sprint       - Full sprint validation workflow"
	@echo "  make validate-quick        - Show requirements for current changes"
	@echo "  make validate-ci           - Show CI validation checklist"
	@echo "  make validate-metrics      - Display recorded metrics"
	@echo "  make validate-dashboard    - Generate dashboard from results"
	@echo "  make validate-rubrics      - Generate rubric templates"
	@echo "  make validate-rules        - Validate rules configuration"
	@echo "  make validate-sync         - Sync rules between directories"
	@echo "  make validate-help         - Detailed validation help"
	@echo ""
	@echo "📋 TASK MANAGEMENT"
	@echo "  make tasks                 - List all tasks"
	@echo "  make generate-tasks        - Generate from roadmap"
	@echo "  make sync-tasks            - Sync with roadmap"
	@echo ""
	@echo "🧪 DEVELOPMENT COMMANDS"
	@echo "  make test                  - Run all tests (pytest)"
	@echo "  make lint                  - Run linters (ruff, mypy)"
	@echo "  make install               - Install dependencies"
	@echo "  make dev-setup             - Setup development environment"
	@echo ""
	@echo "🧹 CLEANUP COMMANDS"
	@echo "  make clean                 - Clean all generated files"
	@echo "  make clean-validation      - Clean validation artifacts"
	@echo ""
	@echo "📚 NATURAL LANGUAGE EXAMPLES:"
	@echo "  make todo what's next           - Get suggestions"
	@echo "  make todo simulate cerebellum   - Simulate cerebellum"
	@echo "  make todo brain status          - Check brain status"
	@echo "  make todo validate foundation   - Validate domain"
	@echo "  make todo train model --stage 1 - Train stage 1"
	@echo "  make todo deploy to gcp         - Deploy to Google Cloud"
	@echo "  make todo plan new task         - Start planning"
	@echo "  make todo workflow new_feature  - Feature workflow"
	@echo ""
	@echo "🧠 BRAIN COMPONENTS: cerebellum, cortex, hippocampus, basal_ganglia,"
	@echo "                     morphogen, e8, cognitive, memory, motor, sensory"
	@echo ""
	@echo "Type 'make todo help' for complete TODO system documentation"

# ==========================================
# TODO SYSTEM - Natural Language Interface
# ==========================================

# Main TODO entry point - accepts natural language
todo:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		python todo.py; \
	else \
		python todo.py $(filter-out $@,$(MAKECMDGOALS)); \
	fi

# Task planning shortcut
plan:
	@python todo.py plan new task

# Work on task
work:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		python todo.py work on task; \
	else \
		python todo.py work on $(filter-out $@,$(MAKECMDGOALS)); \
	fi

# Track progress
track:
	@python todo.py track progress

# Review completed work
review:
	@python todo.py review completed

# Daily standup workflow
standup:
	@python todo.py workflow daily_standup

# Sprint review workflow
sprint:
	@python todo.py workflow sprint_review

# List tasks
tasks:
	@python state/todo/core/tasks_launcher.py list

# Generate tasks from roadmap
generate-tasks:
	@python state/todo/core/tasks_launcher.py generate --stage 1

# Sync tasks with roadmap
sync-tasks:
	@python state/todo/core/tasks_launcher.py sync --update-status

# ==========================================
# BRAIN & SIMULATION COMMANDS
# ==========================================

# Run brain simulation
simulate:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		python todo.py run brain simulation; \
	else \
		python todo.py simulate $(filter-out $@,$(MAKECMDGOALS)); \
	fi

# Brain status
brain-status:
	@python todo.py brain status

# Brain list components
brain-list:
	@python todo.py brain list

# Brain analyze
brain-analyze:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		python todo.py brain analyze; \
	else \
		python todo.py brain analyze $(filter-out $@,$(MAKECMDGOALS)); \
	fi

# Brain test
brain-test:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		python todo.py brain test; \
	else \
		python todo.py brain test $(filter-out $@,$(MAKECMDGOALS)); \
	fi

# Brain profile
brain-profile:
	@python todo.py brain profile

# Brain visualize
brain-visualize:
	@python todo.py brain visualize

# Brain orchestrate
brain-orchestrate:
	@python todo.py brain orchestrate status

# Brain startup
brain-startup:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		python todo.py brain startup; \
	else \
		python todo.py brain startup $(filter-out $@,$(MAKECMDGOALS)); \
	fi

# ==========================================
# TRAINING COMMANDS
# ==========================================

# Start training
train:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		python todo.py train model; \
	else \
		python todo.py train $(filter-out $@,$(MAKECMDGOALS)); \
	fi

# Training status
training-status:
	@python todo.py training status

# ==========================================
# DEPLOYMENT COMMANDS
# ==========================================

# Deploy to platform
deploy:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		python todo.py deploy local; \
	else \
		python todo.py deploy to $(filter-out $@,$(MAKECMDGOALS)); \
	fi

# Deployment status
deployment-status:
	@python todo.py deployment status

# ==========================================
# DOCUMENTATION COMMANDS
# ==========================================

# Generate documentation
docs:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		python todo.py generate docs; \
	else \
		python todo.py $(filter-out $@,$(MAKECMDGOALS)) docs; \
	fi

# Update README
update-readme:
	@python todo.py update readme

# ==========================================
# BENCHMARKING COMMANDS
# ==========================================

# Run benchmarks
benchmark:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		python todo.py benchmark performance; \
	else \
		python todo.py benchmark $(filter-out $@,$(MAKECMDGOALS)); \
	fi

# Compare metrics
compare-metrics:
	@python todo.py compare metrics

# Main validation entry point - supports domain argument
validate:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "🚀 Starting Interactive Validation Guide..."; \
		echo "📝 This will show what needs validation (no auto-validation)"; \
		python state/todo/core/validate_launcher.py sprint; \
	else \
		domain="$$(echo '$(filter-out $@,$(MAKECMDGOALS))' | tr ' ' '_')"; \
		echo "🎯 Showing validation requirements for: $$domain"; \
		python state/todo/core/validate_launcher.py verify --domain "$$domain"; \
	fi

# Catch additional arguments for validate target
%:
	@:

# Full sprint validation guide
validate-sprint:
	@echo "📋 Starting Interactive Sprint Validation Guide..."
	@echo "⚠️ You will need to perform all validation manually"
	@python state/todo/core/validate_launcher.py sprint

# Show validation requirements for current changes or specific domain
validate-quick:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "⚡ Showing Validation Requirements for Current Changes..."; \
		python state/todo/core/validate_launcher.py validate; \
	else \
		domain="$$(echo '$(filter-out $@,$(MAKECMDGOALS))' | tr ' ' '_')"; \
		echo "🎯 Showing Requirements for domain: $$domain"; \
		python state/todo/core/validate_launcher.py validate --domain "$$domain"; \
	fi

# CI validation checklist (shows what needs validation)
validate-ci:
	@echo "🤖 Showing CI Validation Checklist..."
	@echo "📝 Manual validation required for all items"
	@python state/todo/core/validate_launcher.py ci

# Generate validation dashboard from manual validation results
validate-dashboard:
	@echo "📊 Generating Dashboard from Manual Validation Results..."
	@python state/todo/core/validate_launcher.py dashboard
	@echo "Dashboard available at: state/tasks/validation/dashboards/validation_dashboard.html"

# Display validation metrics from manual runs
validate-metrics:
	@echo "📈 Displaying Metrics from Manual Validation Runs..."
	@python state/todo/core/validate_launcher.py metrics

# Generate rubric templates for manual validation
validate-rubrics:
	@echo "📝 Generating Rubric Templates for Manual Validation..."
	@python state/todo/core/validate_launcher.py rubric --action generate

# Validate rules index
validate-rules:
	@echo "📜 Validating Rules Index..."
	@python state/todo/core/validate_launcher.py rules

# Sync rules between cursor and quark
validate-sync:
	@echo "🔄 Syncing Rules..."
	@python state/todo/core/validate_launcher.py rules --action sync

# Clean validation artifacts (preserve evidence)
clean-validation:
	@echo "🧹 Cleaning validation artifacts (preserving evidence)..."
	@find state/tasks/validation -name "*.pyc" -delete
	@find state/tasks/validation -name "__pycache__" -type d -delete
	@echo "✅ Cleanup complete"

# Run tests
test:
	@echo "🧪 Running tests..."
	@pytest tests/ -v

# Run linters
lint:
	@echo "🔍 Running linters..."
	@ruff check .
	@mypy --strict .

# Clean all generated files
clean:
	@echo "🧹 Cleaning generated files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	@pip install -r requirements.txt
	@echo "✅ Installation complete"

# Development setup
dev-setup: install
	@echo "🛠️ Setting up development environment..."
	@pip install -r requirements-dev.txt
	@pre-commit install
	@echo "✅ Development setup complete"

# Run full validation pipeline
validate-full: validate-ci validate-dashboard validate-metrics
	@echo "✅ Full validation pipeline complete"

# Help for validation commands
validate-help:
	@echo "Quark Validation Guide Commands:"
	@echo ""
	@echo "Interactive Guides (Manual Validation Required):"
	@echo "  make validate         - Interactive sprint guide"
	@echo "  make validate-sprint  - Full sprint workflow guide"
	@echo ""
	@echo "Show Requirements:"
	@echo "  make validate-quick   - Show what needs validation"
	@echo "  make validate-ci      - Show CI checklist"
	@echo ""
	@echo "Reports (From Manual Results):"
	@echo "  make validate-dashboard - Generate HTML dashboard"
	@echo "  make validate-metrics   - Show recorded metrics"
	@echo "  make validate-rubrics   - Generate rubric templates"
	@echo ""
	@echo "⚠️ NOTE: All validation is manual - no auto-validation"
