#!/bin/bash -e
# Borrowed from FAIR

# Run this script at project root by "./dev/linter.sh" before you commit

{
  V=$(black --version|cut '-d ' -f3)
  code='import distutils.version; assert "19.3" < distutils.version.LooseVersion("'$V'")'
  python -c "${code}" 2> /dev/null
} || {
  echo "Linter requires black 19.3b0 or higher!"
  exit 1
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR=$(dirname "${DIR}")

echo "Running isort..."
# isort -y -sp "${DIR}"
isort "${DIR}"

echo "Running black..."
black "${DIR}"

echo "Running flake..."
flake8 "${DIR}"
