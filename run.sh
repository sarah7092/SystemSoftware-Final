#!/usr/bin/env bash
set -euo pipefail

# Directory containing compiled binaries.
BIN_DIR="bin"
TRAINER="${BIN_DIR}/trainer"

# Directory for log and model files.
LOG_DIR="logs"

# Training and test datasets.
TRAIN_CSV="data/train.csv"
TEST_CSV="data/test.csv"

mkdir -p "$LOG_DIR"

# Build binaries if trainer is missing.
if [[ ! -x "$TRAINER" ]]; then
  echo "[run] Trainer binary not found, building..."
  ./build.sh
fi

# Location for model parameters (used by backward_layer).
export MODEL_FILE="${MODEL_FILE:-logs/model_params.txt}"

# --------------------------------------------------------------------
# progress_bar current total
#   - Render a simple textual progress bar on a single line.
#   - This function is fully implemented.
# --------------------------------------------------------------------
progress_bar() {
  local current=$1
  local total=$2
  local width=40

  if [[ "$total" -le 0 ]]; then
    printf "\r[progress] processing..."
    return
  fi

  local percent=$(( 100 * current / total ))
  local filled=$(( width * current / total ))
  local empty=$(( width - filled ))

  printf "\r["
  for ((i=0; i<filled; i++)); do printf "#"; done
  for ((i=0; i<empty; i++)); do printf "."; done
  printf "] %3d%% (%d/%d)" "$percent" "$current" "$total"
}

# --------------------------------------------------------------------
# run_phase phase csv log_file err_file
#
# Run one phase ("train" or "test"):
#   - Check that the CSV exists.
#   - Count non-empty lines to know how many samples to expect.
#   - Run trainer with BACKWARD_MODE set to the phase.
#   - Capture stdout into a log file and use SAMPLE lines for progress.
#   - After completion, extract a SUMMARY line and print a short summary.
# --------------------------------------------------------------------
run_phase() {
  local phase="$1"      # "train" or "test"
  local csv="$2"
  local log_file="$3"
  local err_file="$4"

  if [[ ! -f "$csv" ]]; then
    echo "[run] CSV file not found for phase '$phase': $csv" >&2
    return 1
  fi

  # Count total non-empty lines in the CSV.
  local total_lines
  total_lines=$(grep -cve '^\s*$' "$csv" || echo 0)

  echo "[run] Phase: $phase, file: $csv"
  echo "[run] Total lines: $total_lines"
  echo "[run] Logs: $log_file, errors: $err_file"

  local SAMPLES_PROCESSED=0

  # Pipeline:
  #   trainer stdout  -> tee (log_file + progress loop)
  #   trainer stderr  -> err_file
  #
  # The logger prints lines starting with "SAMPLE" for each sample.
  BACKWARD_MODE="$phase" "$TRAINER" "$csv" 2> "$err_file" | \
  tee "$log_file" | \
  while IFS= read -r line; do
    if [[ "$line" == SAMPLE* ]]; then
      SAMPLES_PROCESSED=$((SAMPLES_PROCESSED + 1))
      progress_bar "$SAMPLES_PROCESSED" "$total_lines"
    fi
  done

  echo

  # ------------------------------------------------------------------
  # TODO: Extract and print phase summary from "$log_file".
  #
  # The logger writes a final line of the form:
  #   SUMMARY <samples> <avg_loss> <avg_yhat>
  #
  # 1) Find the last such line in "$log_file" and store it in 'summary'.
  #    If no such line exists, 'summary' should be empty.
  #
  # 2) If 'summary' is non-empty:
  #      - Parse it into variables: samples, avg_loss, avg_yhat.
  #        (The first field is the literal word "SUMMARY".)
  #      - Print:
  #          [run] Phase '<phase>' summary: samples=<samples> avg_loss=<avg_loss> avg_yhat=<avg_yhat>
  #
  #    Otherwise:
  #      - Print:
  #          [run] Phase '<phase>' summary: (no SUMMARY line found)
  # ------------------------------------------------------------------
  local summary=""
  # TODO: summary=...
  summary=$(grep "^SUMMARY" "$log_file" | tail -n 1)

  if [[ -n "$summary" ]]; then
    local _tag samples avg_loss avg_yhat
    # TODO: read -r ...
    read -r _tag samples avg_loss avg_yhat <<< "$summary"

    echo "[run] Phase '$phase' summary: samples=$samples avg_loss=$avg_loss avg_yhat=$avg_yhat"
  else
    echo "[run] Phase '$phase' summary: (no SUMMARY line found)"
  fi

  echo "[run] Phase '$phase' finished."
  echo "[run] Final logs: $log_file"
}

# -------- PRE-TRAIN TEST (baseline) --------
run_phase "test" "$TEST_CSV" \
  "${LOG_DIR}/pre-test.log" \
  "${LOG_DIR}/pre-test.err"

# -------- TRAINING --------
run_phase "train" "$TRAIN_CSV" \
  "${LOG_DIR}/train-train.log" \
  "${LOG_DIR}/train-train.err"

# -------- POST-TRAIN TEST --------
run_phase "test" "$TEST_CSV" \
  "${LOG_DIR}/post-test.log" \
  "${LOG_DIR}/post-test.err"
