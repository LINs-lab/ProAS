# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

_TRANSFORMERS_AUTOCONFIG_VLLM_PATCHED = False


def patch_transformers_autoconfig_for_vllm() -> None:
    """Allow vLLM's optional config registrations to be idempotent.

    vLLM 0.9.x registers ``aimv2`` while importing its Ovis config. Recent
    Transformers releases already include that key, so a second registration
    raises ``ValueError`` before rollout initialization can finish.
    """
    global _TRANSFORMERS_AUTOCONFIG_VLLM_PATCHED
    if _TRANSFORMERS_AUTOCONFIG_VLLM_PATCHED:
        return

    try:
        from transformers.models.auto.configuration_auto import AutoConfig

        original_register = AutoConfig.register

        def register(model_type, config, exist_ok=False):
            if model_type == "aimv2":
                exist_ok = True
            return original_register(model_type, config, exist_ok=exist_ok)

        AutoConfig.register = staticmethod(register)  # type: ignore[method-assign]
        _TRANSFORMERS_AUTOCONFIG_VLLM_PATCHED = True
    except Exception:
        pass
