#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

MODE="full"
case "${1:-}" in
    "") ;;
    --browser-only) MODE="browser" ;;
    -h|--help)
        echo "Usage: $0 [--browser-only]"
        echo "  --browser-only  install the ROOT-free data browser/export environment"
        exit 0
        ;;
    *)
        echo "Unknown option: $1" >&2
        echo "Usage: $0 [--browser-only]" >&2
        exit 2
        ;;
esac

DATA_VERSION="v1.0.0"
DATA_BASENAME="HFIRBG_public_data_${DATA_VERSION}"
DATA_ARCHIVE="${REPO_ROOT}/data/${DATA_BASENAME}.tar.gz"
DATA_URL="https://github.com/BlaineHeffron/HFIR_BG_Analysis/releases/download/data-${DATA_VERSION}/${DATA_BASENAME}.tar.gz"
DATA_SHA256="037dfce3383a7b86d40772a45253423c0e63f1d803eba246279cccb124c9b2c4"

ARXIV_ID="2607.05834v1"
PAPER_ARCHIVE="${REPO_ROOT}/data/arxiv_${ARXIV_ID}_source.tar.gz"
PAPER_DIR="${REPO_ROOT}/data/arxiv_2607.05834"
PAPER_URL="https://arxiv.org/src/${ARXIV_ID}"
PAPER_SHA256="9892ba7a090400d3c2c6f9189b23dfd41415e2978b161b22b3dee56ff69870e9"

download() {
    local url="$1"
    local destination="$2"
    if [[ -s "$destination" ]]; then
        echo "Using existing download: $destination"
        return
    fi
    echo "Downloading $url"
    if ! curl -fL --retry 3 --retry-all-errors --connect-timeout 15 \
        --speed-limit 1024 --speed-time 30 -o "$destination" "$url"; then
        echo "Retrying download over IPv4..."
        curl -4 -fL --retry 3 --retry-all-errors --connect-timeout 15 \
            --speed-limit 1024 --speed-time 30 -o "$destination" "$url"
    fi
}

verify_checksum() {
    python3 - "$1" "$2" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = sys.argv[2]
digest = hashlib.sha256()
with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
actual = digest.hexdigest()
if actual != expected:
    raise SystemExit(f"checksum mismatch for {path}: expected {expected}, got {actual}")
print(f"Checksum verified: {path}")
PY
}

mkdir -p data
download "$DATA_URL" "$DATA_ARCHIVE"
if ! verify_checksum "$DATA_ARCHIVE" "$DATA_SHA256"; then
    echo "Discarding the incomplete or invalid public-data download."
    rm -f "$DATA_ARCHIVE"
    download "$DATA_URL" "$DATA_ARCHIVE"
    verify_checksum "$DATA_ARCHIVE" "$DATA_SHA256"
fi

if [[ ! -f "data/${DATA_BASENAME}/HFIRBG.db" ]]; then
    echo "Extracting public spectra and database..."
    tar -xzf "$DATA_ARCHIVE" -C data
fi

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from portable defaults."
fi

set -a
source .env
set +a
python3 scripts/check_public_data_setup.py --sanitize-database-path

if [[ "$MODE" == "full" ]]; then
    download "$PAPER_URL" "$PAPER_ARCHIVE"
    if ! verify_checksum "$PAPER_ARCHIVE" "$PAPER_SHA256"; then
        echo "Discarding the incomplete or invalid arXiv download."
        rm -f "$PAPER_ARCHIVE"
        download "$PAPER_URL" "$PAPER_ARCHIVE"
        verify_checksum "$PAPER_ARCHIVE" "$PAPER_SHA256"
    fi
    if [[ ! -f "${PAPER_DIR}/anc/UNFOLDING_RESULTS_README.md" ]]; then
        echo "Extracting official paper source and ancillary results..."
        mkdir -p "$PAPER_DIR"
        tar -xf "$PAPER_ARCHIVE" -C "$PAPER_DIR"
    fi
fi

if [[ "$MODE" == "full" && "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]]; then
    echo
    echo "Data setup complete. The PyPI ROOT wheel is only available for Linux x86-64."
    echo "Create the portable analysis environment with:"
    echo "  conda env create -f environment.yml"
    echo "  conda activate hfir-bg-analysis"
    exit 0
fi

if [[ ! -x .venv/bin/python ]]; then
    if ! python3 -m venv .venv; then
        echo "The OS venv package is unavailable; creating a pip-less environment."
        python3 -m venv --without-pip .venv
    fi
fi

if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    GET_PIP="${REPO_ROOT}/data/get-pip.py"
    download "https://bootstrap.pypa.io/get-pip.py" "$GET_PIP"
    .venv/bin/python "$GET_PIP"
fi

REQUIREMENTS_FILE="requirements.txt"
if [[ "$MODE" == "browser" ]]; then
    REQUIREMENTS_FILE="webapp/requirements.txt"
fi
.venv/bin/python -m pip install -r "$REQUIREMENTS_FILE"

export MPLBACKEND="${MPLBACKEND:-Agg}"
if [[ "$MODE" == "browser" ]]; then
    .venv/bin/python - <<'PY'
import numpy
import pandas
import plotly
import streamlit

print(f"NumPy {numpy.__version__}; pandas {pandas.__version__}")
print(f"Plotly {plotly.__version__}; Streamlit {streamlit.__version__}")
PY
else
    .venv/bin/python - <<'PY'
import ROOT
import matplotlib
import numba
import numpy
import pandas
import scipy
import uproot

print(f"ROOT {ROOT.gROOT.GetVersion()}")
print(f"NumPy {numpy.__version__}; SciPy {scipy.__version__}; Matplotlib {matplotlib.__version__}")
print(f"Numba {numba.__version__}; pandas {pandas.__version__}; uproot {uproot.__version__}")
PY
fi

echo
if [[ "$MODE" == "browser" ]]; then
    echo "Browser setup complete. Start it with:"
    echo "  ./scripts/run_data_browser.sh"
    echo "Or export data without the UI with:"
    echo "  .venv/bin/python scripts/export_public_data.py --help"
else
    echo "Full analysis setup complete. In each new shell run:"
    echo "  source .env"
    echo "Then generate the public analysis products with:"
    echo "  .venv/bin/python scripts/public_analysis.py all"
    echo "Or browse all released runs and spectra with:"
    echo "  ./scripts/run_data_browser.sh"
fi
