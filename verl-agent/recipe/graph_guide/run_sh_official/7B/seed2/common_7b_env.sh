#!/usr/bin/env bash

discover_jdk_home() {
  local candidate=""

  if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/javac" ]]; then
    echo "$JAVA_HOME"
    return 0
  fi

  if command -v javac >/dev/null 2>&1; then
    candidate="$(readlink -f "$(command -v javac)")"
    candidate="$(cd "$(dirname "$candidate")/.." && pwd)"
    if [[ -x "${candidate}/bin/javac" ]]; then
      echo "$candidate"
      return 0
    fi
  fi

  local candidates=()
  local path=""
  for path in \
    /usr/lib/jvm/java-17-openjdk-amd64 \
    /usr/lib/jvm/java-11-openjdk-amd64 \
    /usr/lib/jvm/default-java; do
    candidates+=("$path")
  done

  while IFS= read -r path; do
    candidates+=("$path")
  done < <(
    {
      compgen -G "/usr/lib/jvm/java-*-openjdk-*" || true
      compgen -G "/usr/lib/jvm/jdk-*" || true
      compgen -G "/usr/java/*" || true
      compgen -G "/opt/java/*" || true
      compgen -G "/opt/jdk/*" || true
      compgen -G "/usr/local/jdk*" || true
      compgen -G "/usr/local/java*" || true
    } | awk 'NF && !seen[$0]++'
  )

  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}/bin/javac" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

setup_verl_7b_env() {
  local env_name="$1"
  local require_java="${2:-auto}"

  WORK_DIR=/mnt/public/dingbowen/project_sft_rl_agenetic_training/verl-agent
  echo "WORK_DIR=$WORK_DIR"
  cd "$WORK_DIR"
  export PYTHONPATH="$WORK_DIR:$WORK_DIR/verl:${PYTHONPATH:-}"

  VENV_DIR="/mnt/public/dingbowen/uv_envs/verl-agent/${env_name}"
  if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "ERROR: venv activate script not found: ${VENV_DIR}/bin/activate" >&2
    exit 1
  fi
  source "${VENV_DIR}/bin/activate"

  local site_packages
  site_packages=$(find "${VENV_DIR}/lib" -maxdepth 2 -type d -name site-packages 2>/dev/null | head -n 1)
  if [[ -n "$site_packages" ]]; then
    export PYTHONPATH="${site_packages}:${PYTHONPATH:-}"
  fi

  python_is_usable() {
    local candidate="$1"
    [[ -n "$candidate" && -x "$candidate" ]] || return 1
    "$candidate" - <<'PY' >/dev/null 2>&1
import encodings
import hydra
import pandas
import torch
import ray
import verl
import agent_system
PY
  }

  local expected_python="${VENV_DIR}/bin/python3"
  if ! python_is_usable "$expected_python"; then
    echo "WARN: ${expected_python} is not usable; trying to repair venv python symlinks" >&2
    local repaired_python=""
    local py_minor=""
    if [[ -L "${VENV_DIR}/bin/python" ]]; then
      py_minor=$(readlink "${VENV_DIR}/bin/python" | grep -oE 'python3\.[0-9]+' | head -n 1 | sed 's/python//')
    fi
    py_minor=${py_minor:-3.12}

    local python_candidates=()
    while IFS= read -r py_candidate; do
      python_candidates+=("$py_candidate")
    done < <(
      {
        command -v uv >/dev/null 2>&1 && uv python find "$py_minor" 2>/dev/null || true
        find /root/.local/share/uv/python -path "*/bin/python${py_minor}" -type f 2>/dev/null || true
        command -v python3 2>/dev/null || true
        echo /usr/bin/python3
        echo /usr/local/bin/python3
      } | awk 'NF && !seen[$0]++'
    )

    for py_candidate in "${python_candidates[@]}"; do
      if python_is_usable "$py_candidate"; then
        ln -sfn "$py_candidate" "${VENV_DIR}/bin/python"
        ln -sfn python "${VENV_DIR}/bin/python3"
        ln -sfn python "${VENV_DIR}/bin/python${py_minor}"
        hash -r
        repaired_python="$py_candidate"
        break
      fi
    done

    if [[ -z "$repaired_python" && -x "$(command -v uv 2>/dev/null)" ]]; then
      echo "No cached usable Python ${py_minor} found; trying: uv python install ${py_minor}" >&2
      uv python install "$py_minor" || true
      local uv_python
      uv_python=$(uv python find "$py_minor" 2>/dev/null || true)
      if python_is_usable "$uv_python"; then
        ln -sfn "$uv_python" "${VENV_DIR}/bin/python"
        ln -sfn python "${VENV_DIR}/bin/python3"
        ln -sfn python "${VENV_DIR}/bin/python${py_minor}"
        hash -r
        repaired_python="$uv_python"
      fi
    fi

    if [[ -z "$repaired_python" ]]; then
      echo "ERROR: could not repair ${VENV_DIR}/bin/python3" >&2
      ls -lah "${VENV_DIR}/bin"/python* >&2 || true
      exit 1
    fi
    echo "Repaired venv python: ${VENV_DIR}/bin/python -> ${repaired_python}" >&2
  fi

  PYTHON_BIN="${VENV_DIR}/bin/python3"
  if ! python_is_usable "$PYTHON_BIN"; then
    echo "ERROR: expected usable Python at ${PYTHON_BIN}" >&2
    exit 1
  fi
  export PYTHON_BIN
  echo "PYTHON_BIN=$PYTHON_BIN"

  if [[ "$require_java" == "auto" ]]; then
    case "$env_name" in
      weshop|webshop)
        require_java=true
        ;;
      *)
        require_java=false
        ;;
    esac
  fi

  local discovered_java_home=""
  if [[ "$require_java" == "true" ]]; then
    if [[ -n "${JAVA_HOME:-}" && ! -x "${JAVA_HOME}/bin/javac" ]]; then
      echo "WARN: JAVA_HOME=${JAVA_HOME} does not contain bin/javac; trying to auto-discover a JDK" >&2
    fi
    if ! discovered_java_home="$(discover_jdk_home)"; then
      echo "ERROR: javac not found. Set JAVA_HOME to a valid JDK root or install a JDK." >&2
      echo "Checked JAVA_HOME, command -v javac, /usr/lib/jvm, /usr/java, /opt/java, /opt/jdk, and /usr/local." >&2
      exit 1
    fi
    export JAVA_HOME="$discovered_java_home"
    export PATH="$JAVA_HOME/bin:$PATH"
    echo "JAVA_HOME=$JAVA_HOME"
    java -version 2>&1 | head -n 1 || true
    javac -version 2>&1 | head -n 1 || true
  else
    echo "Skipping JDK discovery for env_name=$env_name"
  fi

  local python_include_dir=""
  python_include_dir=$("$PYTHON_BIN" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("INCLUDEPY") or "")
PY
)
  if [[ -z "$python_include_dir" || ! -f "${python_include_dir}/Python.h" ]]; then
    echo "ERROR: Python.h not found under ${python_include_dir:-<unknown>}." >&2
    echo "vLLM/Triton needs the matching Python development headers when actor_rollout_ref.rollout.enforce_eager=False." >&2
    echo "Install them on this machine, for example: apt-get update && apt-get install -y python3.10-dev" >&2
    exit 1
  fi
  echo "Python.h=${python_include_dir}/Python.h"
}
