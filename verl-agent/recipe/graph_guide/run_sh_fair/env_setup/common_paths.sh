#!/usr/bin/env bash

if [[ -n "${VERL_COMMON_PATHS_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
export VERL_COMMON_PATHS_LOADED=1

ENV_SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH_FAIR_ROOT="$(cd -- "${ENV_SETUP_DIR}/.." && pwd)"
GRAPH_GUIDE_ROOT="$(cd -- "${RUN_SH_FAIR_ROOT}/.." && pwd)"
VERL_AGENT_ROOT="$(cd -- "${GRAPH_GUIDE_ROOT}/../.." && pwd)"
PROAS_ROOT="$(cd -- "${VERL_AGENT_ROOT}/.." && pwd)"

export ENV_SETUP_DIR RUN_SH_FAIR_ROOT GRAPH_GUIDE_ROOT VERL_AGENT_ROOT PROAS_ROOT

export ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"
export VERL_DATA_ROOT="${VERL_DATA_ROOT:-${GRAPH_GUIDE_ROOT}/data}"
export VERL_OUTPUT_ROOT="${VERL_OUTPUT_ROOT:-${GRAPH_GUIDE_ROOT}/outputs}"
export VERL_TENSORBOARD_ROOT="${VERL_TENSORBOARD_ROOT:-${GRAPH_GUIDE_ROOT}/tensorboard_fair}"
export VERL_MODEL_ROOT="${VERL_MODEL_ROOT:-${GRAPH_GUIDE_ROOT}/models}"
export VERL_VENV_ROOT="${VERL_VENV_ROOT:-${VERL_AGENT_ROOT}/.venvs}"
export WEBSHOP_VENV_DIR="${WEBSHOP_VENV_DIR:-${VERL_VENV_ROOT}/webshop}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${HOME}/.local/share/uv/python}"
export ALFWORLD_LIBSTDCXX_DIR="${ALFWORLD_LIBSTDCXX_DIR:-}"
export MODEL_PATH_1_5B="${MODEL_PATH_1_5B:-${VERL_MODEL_ROOT}/Qwen2.5-1.5B-Instruct}"
export MODEL_PATH_7B="${MODEL_PATH_7B:-${VERL_MODEL_ROOT}/Qwen2.5-7B-Instruct}"
export WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-${VERL_AGENT_ROOT}/agent_system/environments/env_package/webshop/webshop/data}"
export WEBSHOP_SEARCH_ENGINE_DIR="${WEBSHOP_SEARCH_ENGINE_DIR:-${VERL_AGENT_ROOT}/agent_system/environments/env_package/webshop/webshop/search_engine}"
