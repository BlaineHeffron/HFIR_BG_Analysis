#!/bin/bash
# Run GeCollimatorUnfolder for the six paper spectra.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${HFIRBG_REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"
DATA_DIR="${HFIRBG_UNFOLD_DATA:-${REPO_ROOT}/data}"
P2X_BIN="${P2X_BIN:-}"
CASE_NAME="isotropic"
OVERWRITE=0

usage() {
    cat <<'EOF'
Usage: run_all_unfolds.sh [options]

Options:
  --case <name>              Scenario name. Defaults to "isotropic".
  --migration-matrix <path>  Explicit migration matrix ROOT file.
  --data-dir <path>          Input data directory.
  --out-dir <path>           Output directory for unfolded ROOT files.
  --cfg-dir <path>           Directory for generated cfg files.
  --p2x-bin <path>           Path to P2x_Analyze.
  --overwrite                Re-run even if output file already exists.
  -h, --help                 Show this help text.

Defaults:
  isotropic -> scripts/private/migration_matrix.root, analysis/unfold/sumita
  front     -> scripts/private/migration_matrix_front.root, analysis/unfold/front
EOF
}

MIGRATION_MATRIX=""
OUT_DIR=""
CFG_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --case)
            CASE_NAME="$2"
            shift 2
            ;;
        --migration-matrix)
            MIGRATION_MATRIX="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --cfg-dir)
            CFG_DIR="$2"
            shift 2
            ;;
        --p2x-bin)
            P2X_BIN="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$MIGRATION_MATRIX" ]]; then
    case "$CASE_NAME" in
        isotropic)
            MIGRATION_MATRIX="${REPO_ROOT}/scripts/private/migration_matrix.root"
            ;;
        front)
            MIGRATION_MATRIX="${REPO_ROOT}/scripts/private/migration_matrix_front.root"
            ;;
        *)
            echo "No default migration matrix for case '$CASE_NAME'; pass --migration-matrix." >&2
            exit 1
            ;;
    esac
fi

if [[ -z "$OUT_DIR" ]]; then
    case "$CASE_NAME" in
        isotropic)
            OUT_DIR="${REPO_ROOT}/analysis/unfold/sumita"
            ;;
        *)
            OUT_DIR="${REPO_ROOT}/analysis/unfold/${CASE_NAME}"
            ;;
    esac
fi

if [[ -z "$CFG_DIR" ]]; then
    CFG_DIR="${OUT_DIR}/cfg"
fi

INPUT_LINK_DIR="${OUT_DIR}/input_links"

FNAMES=(
    "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN"
    "MIF_BOX_AT_REACTOR_RXOFF"
    "CYCLE461_DOWN_FACING_OVERNIGHT"
    "HB4_DOWN_OVERNIGHT_1"
    "EAST_FACE_18"
    "EAST_FACE_1"
)

mkdir -p "$OUT_DIR" "$CFG_DIR" "$INPUT_LINK_DIR"

if [[ ! -x "$P2X_BIN" ]]; then
    echo "ERROR: set P2X_BIN or pass --p2x-bin with an executable P2x_Analyze path." >&2
    exit 1
fi

if [[ ! -f "$MIGRATION_MATRIX" ]]; then
    echo "ERROR: Migration matrix not found: $MIGRATION_MATRIX" >&2
    exit 1
fi

echo "Case: $CASE_NAME"
echo "Migration matrix: $MIGRATION_MATRIX"
echo "Input data dir: $DATA_DIR"
echo "Output dir: $OUT_DIR"
echo ""

for fname in "${FNAMES[@]}"; do
    INPUT_HIST="${DATA_DIR}/${fname}.root"
    LINK_INPUT="${INPUT_LINK_DIR}/${fname}.root"
    EXPECTED_OUTPUT="${INPUT_LINK_DIR}/${fname}_unfold_results.root"
    FINAL_OUTPUT="${OUT_DIR}/${fname}_unfold_results.root"
    CFG_FILE="${CFG_DIR}/${fname}.cfg"

    if [[ ! -f "$INPUT_HIST" ]]; then
        echo "ERROR: Input file not found: $INPUT_HIST"
        continue
    fi

    if [[ $OVERWRITE -eq 0 && -f "$FINAL_OUTPUT" ]]; then
        echo "=== SKIP (already exists): $fname ==="
        continue
    fi

    ln -sf "$INPUT_HIST" "$LINK_INPUT"
    rm -f "$EXPECTED_OUTPUT"

    cat > "$CFG_FILE" <<EOF
class: "GeCollimatorUnfolder"
migrationFile: "${MIGRATION_MATRIX}"
inputHist: "${LINK_INPUT}"
EHigh: 12000
ELow: 40
EOF

    echo "=== Unfolding (${CASE_NAME}): $fname ==="
    "$P2X_BIN" "$CFG_FILE" 2>&1 | tail -20

    if [[ -f "$EXPECTED_OUTPUT" ]]; then
        mv -f "$EXPECTED_OUTPUT" "$FINAL_OUTPUT"
        echo "  -> Output: $FINAL_OUTPUT"
    else
        echo "  -> ERROR: Output not created!"
    fi
    echo ""
done

echo "=== All unfolding complete (${CASE_NAME}) ==="
ls -la "$OUT_DIR"/*.root 2>/dev/null || true
