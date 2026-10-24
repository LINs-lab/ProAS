#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  clean_official_outputs_except_7b.sh [options] [worker options]

Default behavior:
  Discover every checkpoint root under:
    /mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent/official_outputs

  Skip everything under:
    /mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent/official_outputs/7B

  Then call merge_hf_and_clean_fsdp_artifacts.sh on all discovered roots.
  Layouts like exp/exp/global_step_* and rewardflow_*/checkpoints/global_step_*
  are discovered automatically.

Options handled by this wrapper:
  --base-dir DIR        Override the official_outputs dir.
  --exclude-dir DIR     Exclude a directory tree. Can be repeated.
  --clean-script PATH   Override merge_hf_and_clean_fsdp_artifacts.sh path.
  --list-only           Print discovered checkpoint roots and exit.
  -h, --help            Show this help.

Common worker options you can pass through:
  --dry-run
  --keep-steps 100,150
  --keep-step 150
  --python-bin /path/to/python
  --merger-script /path/to/model_merger.py

Examples:
  # Print all checkpoint roots that would be processed.
  clean_official_outputs_except_7b.sh --list-only

  # Preview all merge and delete actions.
  clean_official_outputs_except_7b.sh --dry-run

  # Actually merge HF weights and clean artifacts.
  clean_official_outputs_except_7b.sh

  # Use the same explicit keep steps for every discovered checkpoint root.
  clean_official_outputs_except_7b.sh --keep-steps 100,150
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_BASE_DIR="/mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent/official_outputs"
DEFAULT_EXCLUDE_DIR="/mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent/official_outputs/7B"

BASE_DIR="$DEFAULT_BASE_DIR"
CLEAN_SCRIPT="${SCRIPT_DIR}/merge_hf_and_clean_fsdp_artifacts.sh"
EXCLUDE_DIRS=("$DEFAULT_EXCLUDE_DIR")
PASSTHROUGH_ARGS=()
LIST_ONLY=0

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

info() {
  echo "[INFO] $*"
}

canonical_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "Directory does not exist: $path"
  (cd "$path" && pwd -P)
}

is_under_exclude() {
  local root="$1"
  local exclude

  for exclude in "${EXCLUDE_DIRS_CANONICAL[@]}"; do
    case "$root" in
      "$exclude"|"$exclude"/*)
        return 0
        ;;
    esac
  done

  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir)
      [[ $# -ge 2 ]] || die "--base-dir requires a value"
      BASE_DIR="$2"
      shift 2
      ;;
    --base-dir=*)
      BASE_DIR="${1#--base-dir=}"
      shift
      ;;
    --exclude-dir)
      [[ $# -ge 2 ]] || die "--exclude-dir requires a value"
      EXCLUDE_DIRS+=("$2")
      shift 2
      ;;
    --exclude-dir=*)
      EXCLUDE_DIRS+=("${1#--exclude-dir=}")
      shift
      ;;
    --clean-script)
      [[ $# -ge 2 ]] || die "--clean-script requires a value"
      CLEAN_SCRIPT="$2"
      shift 2
      ;;
    --clean-script=*)
      CLEAN_SCRIPT="${1#--clean-script=}"
      shift
      ;;
    --list-only)
      LIST_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      PASSTHROUGH_ARGS+=("$@")
      break
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

BASE_DIR="$(canonical_dir "$BASE_DIR")"
[[ -f "$CLEAN_SCRIPT" ]] || die "Clean worker script not found: $CLEAN_SCRIPT"
[[ -x "$CLEAN_SCRIPT" ]] || die "Clean worker script is not executable: $CLEAN_SCRIPT"

EXCLUDE_DIRS_CANONICAL=()
for exclude_dir in "${EXCLUDE_DIRS[@]}"; do
  EXCLUDE_DIRS_CANONICAL+=("$(canonical_dir "$exclude_dir")")
done

mapfile -t CKPT_ROOTS < <(
  while IFS= read -r root; do
    root="$(canonical_dir "$root")"
    if ! is_under_exclude "$root"; then
      printf '%s\n' "$root"
    fi
  done < <(find "$BASE_DIR" -type d -name 'global_step_*' -printf '%h\n' | sort -u) | sort -uV
)

[[ "${#CKPT_ROOTS[@]}" -gt 0 ]] || die "No checkpoint roots found under $BASE_DIR after exclusions"

info "Base dir: $BASE_DIR"
info "Excluded dir tree(s):"
printf '[INFO]   %s\n' "${EXCLUDE_DIRS_CANONICAL[@]}"
info "Discovered ${#CKPT_ROOTS[@]} checkpoint root(s):"
printf '[INFO]   %s\n' "${CKPT_ROOTS[@]}"

if [[ "$LIST_ONLY" -eq 1 ]]; then
  exit 0
fi

"$CLEAN_SCRIPT" "${CKPT_ROOTS[@]}" "${PASSTHROUGH_ARGS[@]}"
