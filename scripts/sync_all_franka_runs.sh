#!/usr/bin/env bash
# Thin wrapper around sync_run_to_gcs.py: walks every logs/train_franka*/*/
# run directory, dispatches it to the right --experiment bucket based on
# its log-root directory name, and prints a summary table at the end.
#
# No Isaac Sim involvement -- plain bash + python3 + gcloud.
#
# Usage:
#   scripts/sync_all_franka_runs.sh [extra args passed through to sync_run_to_gcs.py, e.g. --dry-run]
#
# Continue-on-error: a failure/skip on one run dir does not stop the rest.

set -u  # (deliberately not -e: we want to continue past per-run failures)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYNC_SCRIPT="${REPO_ROOT}/scripts/sync_run_to_gcs.py"

# log-root directory basename -> experiment name
declare -A EXPERIMENT_MAP=(
  [train_franka]="franka-lift-baseline"
  [train_franka_jointdie]="joint-space-die-lift"
  [train_franka_jointcube]="joint-space-die-lift"
  [train_franka_jointdieheavy]="asset-bisect"
  [train_franka_jointdiebig]="asset-bisect"
  [train_franka_jointcubebaked]="asset-bisect"
)

EXTRA_ARGS=("$@")

UPLOADED=0
SKIPPED=0
FAILED=0
declare -a RESULT_ROWS=()

cd "${REPO_ROOT}"

for log_root_path in logs/train_franka*; do
  [ -d "${log_root_path}" ] || continue
  log_root_name="$(basename "${log_root_path}")"
  experiment="${EXPERIMENT_MAP[${log_root_name}]:-}"
  if [ -z "${experiment}" ]; then
    echo "WARN: no --experiment mapping for log root '${log_root_name}', skipping its run dirs"
    continue
  fi

  for run_dir in "${log_root_path}"/*/; do
    [ -d "${run_dir}" ] || continue
    run_dir="${run_dir%/}"  # strip trailing slash
    echo "=================================================================="
    echo "Syncing: ${run_dir}  (experiment=${experiment})"
    echo "=================================================================="

    output="$(python3 "${SYNC_SCRIPT}" --run-dir "${run_dir}" --experiment "${experiment}" --backfill "${EXTRA_ARGS[@]}" 2>&1)"
    rc=$?
    echo "${output}"

    if [ ${rc} -ne 0 ]; then
      FAILED=$((FAILED + 1))
      RESULT_ROWS+=("FAILED   ${run_dir}  (${experiment})")
    elif echo "${output}" | grep -q "^SKIP:"; then
      SKIPPED=$((SKIPPED + 1))
      RESULT_ROWS+=("SKIPPED  ${run_dir}  (${experiment})")
    else
      UPLOADED=$((UPLOADED + 1))
      RESULT_ROWS+=("UPLOADED ${run_dir}  (${experiment})")
    fi
  done
done

echo
echo "=================================================================="
echo "SUMMARY"
echo "=================================================================="
for row in "${RESULT_ROWS[@]}"; do
  echo "  ${row}"
done
echo
echo "Uploaded: ${UPLOADED}   Skipped: ${SKIPPED}   Failed: ${FAILED}"

if [ ${FAILED} -gt 0 ]; then
  exit 1
fi
exit 0
