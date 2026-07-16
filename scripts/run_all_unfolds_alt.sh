#!/bin/bash
# Run the alternate Ge unfolding algorithm for the six paper spectra.

set -euo pipefail

REPO_ROOT="/home/blaine/projects/HFIR_BG_Analysis"
DATA_DIR="/home/blaine/projects/HFIRBG/data"
P2X_BIN="/home/blaine/src/P2x/bin/P2x_Analyze"
CASE_NAME="isotropic"
OVERWRITE=0
REGULARIZATION="1e-4"
STEP_SIZE="1.0"
DELTA_CHISQR="0.005"
MAX_ITERATIONS="4000"

usage() {
    cat <<'EOF'
Usage: run_all_unfolds_alt.sh [options]

Options:
  --case <name>              Scenario name. Defaults to "isotropic".
  --migration-matrix <path>  Explicit migration matrix ROOT file.
  --data-dir <path>          Input data directory.
  --out-dir <path>           Output directory for unfolded ROOT files.
  --cfg-dir <path>           Directory for generated cfg files.
  --p2x-bin <path>           Path to P2x_Analyze.
  --regularization <value>   Smoothness penalty strength.
  --step-size <value>        Iteration step size.
  --delta-chi-sqr <value>    Objective convergence threshold.
  --max-iterations <value>   Maximum Poisson-PGD iterations.
  --overwrite                Re-run even if output file already exists.
  -h, --help                 Show this help text.

Defaults:
  isotropic -> scripts/private/migration_matrix.root, analysis/unfold/poisson_pgd_isotropic
  front     -> scripts/private/migration_matrix_front.root, analysis/unfold/poisson_pgd_front
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
        --regularization)
            REGULARIZATION="$2"
            shift 2
            ;;
        --step-size)
            STEP_SIZE="$2"
            shift 2
            ;;
        --delta-chi-sqr)
            DELTA_CHISQR="$2"
            shift 2
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
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
            OUT_DIR="${REPO_ROOT}/analysis/unfold/poisson_pgd_isotropic"
            ;;
        front)
            OUT_DIR="${REPO_ROOT}/analysis/unfold/poisson_pgd_front"
            ;;
        *)
            OUT_DIR="${REPO_ROOT}/analysis/unfold/poisson_pgd_${CASE_NAME}"
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
    echo "ERROR: P2x_Analyze not found or not executable: $P2X_BIN" >&2
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
echo "Regularization: $REGULARIZATION"
echo "Step size: $STEP_SIZE"
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
class: "GeCollimatorUnfolderAlt"
migrationFile: "${MIGRATION_MATRIX}"
inputHist: "${LINK_INPUT}"
EHigh: 12000
ELow: 40
deltaChiSqr: ${DELTA_CHISQR}
maxIterations: ${MAX_ITERATIONS}
regularization: ${REGULARIZATION}
stepSize: ${STEP_SIZE}
EOF

    echo "=== Alternate unfold (${CASE_NAME}): $fname ==="
    "$P2X_BIN" "$CFG_FILE" 2>&1 | tail -20

    if [[ -f "$EXPECTED_OUTPUT" ]]; then
        mv -f "$EXPECTED_OUTPUT" "$FINAL_OUTPUT"
        echo "  -> Output: $FINAL_OUTPUT"
    else
        echo "  -> ERROR: Output not created!"
    fi
    echo ""
done

echo "=== All alternate unfolding complete (${CASE_NAME}) ==="
ls -la "$OUT_DIR"/*.root 2>/dev/null || true
