#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  merge_hf_and_clean_fsdp_artifacts.sh DIR [DIR ...] [options]
  merge_hf_and_clean_fsdp_artifacts.sh --dir DIR [--dir DIR ...] [options]
  dir=DIR merge_hf_and_clean_fsdp_artifacts.sh [options]

Options:
  --dir DIR                 Checkpoint root containing global_step_* dirs.
                            Can be repeated.
  --keep-steps STEPS        Comma/space separated steps to convert to HF and keep artifacts.
                            Examples: "150", "global_step_150,global_step_200".
                            If omitted, latest_checkpointed_iteration.txt is used.
                            Use "none" to keep no artifacts.
  --keep-step STEP          Add one step to the keep list. Can be repeated.
  --actor-subdir NAME       Actor checkpoint subdir under each global_step_* (default: actor).
  --python-bin PATH         Python executable for model_merger.py and validation.
                            Defaults to $PYTHON_BIN or python3.
  --merger-script PATH      Override model_merger.py path.
  --dry-run                 Print actions without merging or deleting.
  -h, --help                Show this help.

Behavior:
  For each DIR/global_step_*/ACTOR_SUBDIR:
    1. If HF weights are missing but FSDP model_world_* shards exist, merge FSDP to HF.
    2. Verify complete HF weights exist.
    3. For steps not in --keep-steps, delete:
       ACTOR_SUBDIR/{extra_state_world_*,model_world_*,optim_world_*} and global_step_*/data.pt.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_AGENT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

CKPT_ROOT="${dir:-}"
CKPT_ROOTS=()
ACTOR_SUBDIR="actor"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MERGER_SCRIPT="${VERL_AGENT_ROOT}/scripts/model_merger.py"
DRY_RUN=0
KEEP_SPECIFIED=0
KEEP_STEP_TOKENS=()

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

info() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*" >&2
}

normalize_step() {
  local token="$1"
  token="$(printf '%s' "$token" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  token="${token%/}"
  token="${token##*/}"

  if [[ -z "$token" ]]; then
    return 1
  fi

  if [[ "$token" =~ ^global_step_[0-9]+$ ]]; then
    printf '%s\n' "$token"
  elif [[ "$token" =~ ^[0-9]+$ ]]; then
    printf 'global_step_%s\n' "$token"
  else
    return 1
  fi
}

append_keep_steps() {
  local raw="$1"
  local token

  raw="${raw//,/ }"
  for token in $raw; do
    [[ -z "$token" ]] && continue
    KEEP_STEP_TOKENS+=("$token")
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      [[ $# -ge 2 ]] || die "--dir requires a value"
      CKPT_ROOTS+=("$2")
      shift 2
      ;;
    --dir=*)
      CKPT_ROOTS+=("${1#--dir=}")
      shift
      ;;
    dir=*)
      CKPT_ROOT="${1#dir=}"
      shift
      ;;
    --keep-steps)
      [[ $# -ge 2 ]] || die "--keep-steps requires a value"
      KEEP_SPECIFIED=1
      if [[ "$2" != "none" ]]; then
        append_keep_steps "$2"
      fi
      shift 2
      ;;
    --keep-steps=*)
      KEEP_SPECIFIED=1
      value="${1#--keep-steps=}"
      if [[ "$value" != "none" ]]; then
        append_keep_steps "$value"
      fi
      shift
      ;;
    --keep-step)
      [[ $# -ge 2 ]] || die "--keep-step requires a value"
      KEEP_SPECIFIED=1
      KEEP_STEP_TOKENS+=("$2")
      shift 2
      ;;
    --keep-step=*)
      KEEP_SPECIFIED=1
      KEEP_STEP_TOKENS+=("${1#--keep-step=}")
      shift
      ;;
    --actor-subdir)
      [[ $# -ge 2 ]] || die "--actor-subdir requires a value"
      ACTOR_SUBDIR="$2"
      shift 2
      ;;
    --actor-subdir=*)
      ACTOR_SUBDIR="${1#--actor-subdir=}"
      shift
      ;;
    --python-bin)
      [[ $# -ge 2 ]] || die "--python-bin requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --python-bin=*)
      PYTHON_BIN="${1#--python-bin=}"
      shift
      ;;
    --merger-script)
      [[ $# -ge 2 ]] || die "--merger-script requires a value"
      MERGER_SCRIPT="$2"
      shift 2
      ;;
    --merger-script=*)
      MERGER_SCRIPT="${1#--merger-script=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      CKPT_ROOTS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$CKPT_ROOT" ]]; then
  CKPT_ROOTS=("$CKPT_ROOT" "${CKPT_ROOTS[@]}")
fi

[[ "${#CKPT_ROOTS[@]}" -gt 0 ]] || { usage; die "At least one checkpoint dir is required"; }
[[ -f "$MERGER_SCRIPT" ]] || die "model_merger.py not found: $MERGER_SCRIPT"

format_keep_steps() {
  local -n keep_steps_ref="$1"
  local steps=("${!keep_steps_ref[@]}")
  if [[ "${#steps[@]}" -eq 0 ]]; then
    printf 'none'
  else
    printf '%s\n' "${steps[@]}" | sort -V | paste -sd ',' -
  fi
}

hf_weights_complete() {
  local actor_dir="$1"
  "$PYTHON_BIN" - "$actor_dir" <<'PY'
import json
import sys
from pathlib import Path

actor_dir = Path(sys.argv[1])

for name in ("model.safetensors", "pytorch_model.bin"):
    path = actor_dir / name
    if path.is_file() and path.stat().st_size > 0:
        sys.exit(0)

for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
    index_path = actor_dir / index_name
    if not index_path.is_file() or index_path.stat().st_size == 0:
        continue
    try:
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
    except Exception:
        continue
    shard_names = set(weight_map.values())
    if shard_names and all((actor_dir / shard).is_file() and (actor_dir / shard).stat().st_size > 0 for shard in shard_names):
        sys.exit(0)

sys.exit(1)
PY
}

has_fsdp_model_shards() {
  local actor_dir="$1"
  local shards=("${actor_dir}"/model_world_size_*_rank_*.pt)
  [[ "${#shards[@]}" -gt 0 ]]
}

merge_to_hf() {
  local actor_dir="$1"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    info "[dry-run] merge FSDP -> HF: $actor_dir"
    return 0
  fi

  (
    cd "$VERL_AGENT_ROOT"
    "$PYTHON_BIN" "$MERGER_SCRIPT" merge \
      --backend fsdp \
      --local_dir "$actor_dir" \
      --target_dir "$actor_dir"
  )
}

remove_artifacts() {
  local step_dir="$1"
  local actor_dir="$2"
  local artifacts=(
    "${actor_dir}"/extra_state_world_*
    "${actor_dir}"/model_world_*
    "${actor_dir}"/optim_world_*
    "${step_dir}"/data.pt
  )
  local existing=()
  local path

  for path in "${artifacts[@]}"; do
    [[ -e "$path" ]] && existing+=("$path")
  done

  if [[ "${#existing[@]}" -eq 0 ]]; then
    info "No artifacts to delete under $step_dir"
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[INFO] [dry-run] delete artifact: %s\n' "${existing[@]}"
  else
    rm -f -- "${existing[@]}"
    info "Deleted ${#existing[@]} artifact(s) under $step_dir"
  fi
}

shopt -s nullglob

build_keep_steps_for_root() {
  local ckpt_root="$1"
  local -n keep_steps_ref="$2"
  local latest_file
  local latest_step
  local normalized
  local token
  local keep_tokens=("${KEEP_STEP_TOKENS[@]}")

  keep_steps_ref=()

  if [[ "$KEEP_SPECIFIED" -eq 0 ]]; then
    latest_file="${ckpt_root}/latest_checkpointed_iteration.txt"
    if [[ -f "$latest_file" ]]; then
      latest_step="$(sed -n '1{s/^[[:space:]]*//;s/[[:space:]]*$//;p;q}' "$latest_file")"
      if [[ -n "$latest_step" ]]; then
        keep_tokens+=("$latest_step")
      else
        warn "$ckpt_root/latest_checkpointed_iteration.txt is empty; no step will keep artifacts by default"
      fi
    else
      warn "$ckpt_root/latest_checkpointed_iteration.txt not found; no step will keep artifacts by default"
    fi
  fi

  for token in "${keep_tokens[@]}"; do
    if normalized="$(normalize_step "$token")"; then
      keep_steps_ref["$normalized"]=1
    else
      die "Invalid step token: $token"
    fi
  done
}

process_ckpt_root() {
  local ckpt_root="$1"
  local -A keep_steps=()
  local step_dirs=()
  local step_dir
  local step_name
  local actor_dir
  local keep_artifacts
  local hf_ready

  [[ -d "$ckpt_root" ]] || die "Checkpoint dir does not exist: $ckpt_root"

  build_keep_steps_for_root "$ckpt_root" keep_steps
  mapfile -t step_dirs < <(find "$ckpt_root" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' | sort -V)
  [[ "${#step_dirs[@]}" -gt 0 ]] || die "No global_step_* dirs found under $ckpt_root"

  info "Checkpoint dir: $ckpt_root"
  info "Keep artifacts for: $(format_keep_steps keep_steps)"
  info "Actor subdir: $ACTOR_SUBDIR"

  for step_dir in "${step_dirs[@]}"; do
    step_name="$(basename "$step_dir")"
    actor_dir="${step_dir}/${ACTOR_SUBDIR}"

    if [[ ! -d "$actor_dir" ]]; then
      warn "Skip $step_name: actor dir not found: $actor_dir"
      continue
    fi

    keep_artifacts=0
    if [[ -n "${keep_steps[$step_name]:-}" ]]; then
      keep_artifacts=1
    fi

    info "Processing $step_name (keep_artifacts=$keep_artifacts)"

    hf_ready=0
    if hf_weights_complete "$actor_dir"; then
      hf_ready=1
      info "$step_name already has complete HF weights"
    elif has_fsdp_model_shards "$actor_dir"; then
      merge_to_hf "$actor_dir"
      if [[ "$DRY_RUN" -eq 1 ]]; then
        info "[dry-run] skip post-merge HF validation for $step_name"
        hf_ready=1
      elif hf_weights_complete "$actor_dir"; then
        hf_ready=1
        info "$step_name HF weights verified after merge"
      else
        die "$step_name merge finished but complete HF weights were not found in $actor_dir"
      fi
    else
      warn "Skip $step_name: no complete HF weights and no FSDP model_world_size_*_rank_*.pt shards"
    fi

    if [[ "$hf_ready" -eq 1 && "$keep_artifacts" -eq 0 ]]; then
      remove_artifacts "$step_dir" "$actor_dir"
    elif [[ "$keep_artifacts" -eq 1 ]]; then
      info "Keeping artifacts for $step_name"
    fi
  done
}

for ckpt_root in "${CKPT_ROOTS[@]}"; do
  process_ckpt_root "$ckpt_root"
done

info "Done"
