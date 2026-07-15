#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-${repo_root}/.venv/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  python_bin="python3"
fi

cd "${repo_root}"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
exec "${python_bin}" -m streamlit run webapp/app.py "$@"
