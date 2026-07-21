#!/bin/bash
# Generate Python lockfile using pip-tools for reproducible builds
# Usage: ./scripts/generate-lockfile.sh
# Requirements: pip-tools must be installed (pip install pip-tools)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Install pip-tools if not already installed
echo "Ensuring pip-tools is available..."
pip install --quiet pip-tools

# Generate main requirements.lock from requirements.txt
echo "Generating requirements.lock from requirements.txt..."
pip-compile \
  --resolver=backtracking \
  --strip-extras \
  --output-file=requirements.lock \
  requirements.txt

# Generate dev requirements.lock from requirements-dev.txt
echo "Generating requirements-dev.lock from requirements-dev.txt..."
pip-compile \
  --resolver=backtracking \
  --strip-extras \
  --output-file=requirements-dev.lock \
  requirements-dev.txt

echo "✓ Lockfiles generated successfully."
echo "  - requirements.lock (for production)"
echo "  - requirements-dev.lock (for development)"
echo ""
echo "To use these lockfiles, install with:"
echo "  pip install -r requirements.lock"
echo "  pip install -r requirements-dev.lock"
