#!/bin/bash -e
# Borrowed from FAIR

# Run this script at project root by "./dev/linter.sh" before you commit

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR=$(dirname "${DIR}")

echo "Running isort..."
# isort -y -sp "${DIR}"
isort "${DIR}"

echo "Running black..."
black "${DIR}"

echo "Running flake..."
flake8 "${DIR}"
