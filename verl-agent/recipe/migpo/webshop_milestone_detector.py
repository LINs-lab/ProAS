#!/usr/bin/env python3
"""
Standalone milestone detector for WebShop trajectories.

This module provides a reusable interface for checking whether a sequence of
actions/observations has reached the four MIGPO milestones:

1. Search phase: a search results page contains at least one product that
   satisfies all goal constraints (attributes, options availability, price).
2. Detail phase: the agent has opened the detail page of a goal-consistent
   product.
3. Option phase: all goal-required options (e.g., color, size) have been
   correctly selected on the detail page.
4. Purchase phase: the agent purchases a goal-consistent product while all
   constraints remain satisfied.

The detector does not modify the environment. It consumes `(action, state,
next_state, info)` tuples and reports milestone completion events in order.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, Any

_WEBSHOP_ROOT = (
    Path(__file__).resolve().parents[2]
    / "agent_system"
    / "environments"
    / "env_package"
    / "webshop"
    / "webshop"
)
if str(_WEBSHOP_ROOT) not in sys.path:
    sys.path.append(str(_WEBSHOP_ROOT))

from agent_system.environments.env_package.webshop.webshop.web_agent_site.engine.engine import (  # noqa: E501
    load_products,
    parse_action,
)
from agent_system.environments.env_package.webshop.webshop.web_agent_site.engine.goal import (  # noqa: E501
    get_attribute_reward,
    get_option_reward,
)
from agent_system.environments.env_package.webshop.webshop.web_agent_site.utils import (  # noqa: E501
    DEFAULT_FILE_PATH,
    DEFAULT_ATTR_PATH,
)


class MilestonePhase(Enum):
    """Four sequential phases for milestone rewards."""

    SEARCH = auto()
    DETAIL = auto()
    OPTIONS = auto()
    PURCHASE = auto()
    COMPLETE = auto()


@dataclass
class MilestoneResult:
    """Return value for each processed step."""

    phase: MilestonePhase
    achieved: bool
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class MilestoneDetector:
    """
    Stateful detector that walks through four milestone phases.

    Usage:
        detector = MilestoneDetector(goal_dict, product_lookup, price_lookup)
        result = detector.process(action, state, next_state, info)
        if result.achieved:
            print("Milestone reached:", result.phase)
    """

    # 修改为通用的 ASIN 匹配模式，适用于 text 观察模式（ASIN [SEP] 格式）
    ASIN_PATTERN = re.compile(r"\b([A-Z0-9]{10})\b")

    def __init__(
        self,
        goal: Dict[str, Any],
        product_lookup: Dict[str, Dict[str, Any]],
        price_lookup: Dict[str, float],
        log_dir: Optional[str] = None,
    ) -> None:
        self.product_lookup = product_lookup
        self.price_lookup = price_lookup
        self.goal = goal
        self.goal_attr_count = len(goal.get("attributes", []))
        self.goal_option_values = self._normalize_goal_options(goal.get("goal_options"))

        # 日志配置 - 默认启用
        default_log_dir = "logs/milestone_debug"
        self.log_dir = Path(log_dir) if log_dir is not None else Path(default_log_dir)
        self.step_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = None
        self.action_history = []

        # 创建日志目录和文件
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"milestone_debug_{self.session_id}.log"
        self._log_session_init()

        self.reset(goal)

    def reset(self, goal: Optional[Dict[str, Any]] = None) -> None:
        """Reset detector state for a new session."""
        if goal is not None:
            self.goal = goal
            self.goal_attr_count = len(goal.get("attributes", []))
            self.goal_option_values = self._normalize_goal_options(
                goal.get("goal_options")
            )
        self.phase = MilestonePhase.SEARCH
        self.selected_product_asin: Optional[str] = None
        self.matched_option_values: Set[str] = set()
        self.history: List[MilestoneResult] = []
        self.step_count = 0
        self.action_history = []

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def process(
        self,
        action: str,
        state: str,
        next_state: str,
        info: Optional[Dict[str, Any]] = None,
    ) -> MilestoneResult:
        """
        Consume one environment transition and update milestone tracking.

        Args:
            action: string action issued to WebShop (e.g., "search[usb drive]").
            state: observation before the action.
            next_state: observation after the action.
            info: optional info dict returned by the environment.
        """
        if self.phase is MilestonePhase.COMPLETE:
            result = MilestoneResult(
                phase=self.phase,
                achieved=False,
                message="All milestones completed.",
            )
            self.history.append(result)
            self._log_step(action, state, next_state, result)
            return result

        if self.phase is MilestonePhase.SEARCH:
            result = self._process_search_phase(action, next_state)
        elif self.phase is MilestonePhase.DETAIL:
            result = self._process_detail_phase(action)
        elif self.phase is MilestonePhase.OPTIONS:
            result = self._process_options_phase(action, next_state)
        else:  # PURCHASE
            result = self._process_purchase_phase(action, info)

        if result.achieved:
            self._advance_phase()
        self.history.append(result)
        self._log_step(action, state, next_state, result)
        return result

    # -------------------------------------------------------------------------
    # Phase handlers
    # -------------------------------------------------------------------------
    def _process_search_phase(self, action: str, next_state: str) -> MilestoneResult:
        action_name, _ = parse_action(action)
        if action_name.lower() != "search":
            return MilestoneResult(
                phase=self.phase,
                achieved=False,
                message="Waiting for a search action.",
            )

        if "Page 1 (Total results" not in next_state:
            return MilestoneResult(
                phase=self.phase,
                achieved=False,
                message="Search produced no result list yet.",
            )

        candidate_asins = self._extract_asins_from_state(next_state)
        satisfying_asins = [
            asin for asin in candidate_asins if self._product_can_meet_goal(asin)
        ]
        if satisfying_asins:
            return MilestoneResult(
                phase=self.phase,
                achieved=True,
                message="Search results contain goal-consistent product.",
                metadata={"asins": satisfying_asins},
            )
        return MilestoneResult(
            phase=self.phase,
            achieved=False,
            message="No product in the search page satisfies the goal.",
        )

    def _process_detail_phase(self, action: str) -> MilestoneResult:
        action_name, raw_arg = parse_action(action)
        candidate_asin = (raw_arg or "").upper()
        if action_name.lower() != "click" or len(candidate_asin) != 10:
            return MilestoneResult(
                phase=self.phase,
                achieved=False,
                message="Waiting for a product click action.",
            )

        if not self._product_can_meet_goal(candidate_asin):
            return MilestoneResult(
                phase=self.phase,
                achieved=False,
                message=f"Product {candidate_asin} does not satisfy constraints.",
            )

        self.selected_product_asin = candidate_asin
        # If there are no option requirements, mark completion immediately
        if not self.goal_option_values:
            self.phase = MilestonePhase.OPTIONS
        return MilestoneResult(
            phase=self.phase,
            achieved=True,
            message=f"Entered goal-consistent detail page for {candidate_asin}.",
            metadata={"asin": candidate_asin},
        )

    def _process_options_phase(self, action: str, next_state: str) -> MilestoneResult:
        if not self.goal_option_values:
            return MilestoneResult(
                phase=self.phase,
                achieved=True,
                message="No options required for this goal.",
            )

        action_name, action_arg = parse_action(action)
        if action_name.lower() != "click":
            return MilestoneResult(
                phase=self.phase,
                achieved=False,
                message="Waiting for option selection clicks.",
            )

        option_value = self._normalize_text(action_arg or "")
        matched = False
        if option_value in self.goal_option_values:
            matched = True
        elif "you have clicked" in next_state.lower():
            # Parse confirmation text
            clicked_value = next_state.lower().split("you have clicked", 1)[1].strip()
            clicked_value = clicked_value.split(".")[0].strip()
            if clicked_value in self.goal_option_values:
                option_value = clicked_value
                matched = True

        if matched:
            self.matched_option_values.add(option_value)

        if self.goal_option_values.issubset(self.matched_option_values):
            return MilestoneResult(
                phase=self.phase,
                achieved=True,
                message="All required options selected.",
                metadata={
                    "selected_options": sorted(self.matched_option_values),
                },
            )

        return MilestoneResult(
            phase=self.phase,
            achieved=False,
            message="Still waiting for the remaining options.",
            metadata={"selected_options": sorted(self.matched_option_values)},
        )

    def _process_purchase_phase(
        self,
        action: str,
        info: Optional[Dict[str, Any]],
    ) -> MilestoneResult:
        if action.lower() != "click[buy now]":
            return MilestoneResult(
                phase=self.phase,
                achieved=False,
                message='Waiting for "click[buy now]".',
            )

        if not self.selected_product_asin:
            return MilestoneResult(
                phase=self.phase,
                achieved=False,
                message="No product was selected before purchase.",
            )

        final_ok = self._product_fully_satisfies_goal(self.selected_product_asin)

        if info and not final_ok:
            verbose = info.get("verbose")
            if verbose:
                final_ok = (
                    verbose.get("r_att", 0) == 1
                    and verbose.get("r_price", 0) == 1
                    and verbose.get("r_option", 1 if not self.goal_option_values else 0)
                    == 1
                )

        if final_ok:
            return MilestoneResult(
                phase=self.phase,
                achieved=True,
                message="Purchase finished with all constraints satisfied.",
                metadata={"asin": self.selected_product_asin},
            )

        return MilestoneResult(
            phase=self.phase,
            achieved=False,
            message="Purchase did not satisfy all constraints.",
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _advance_phase(self) -> None:
        if self.phase is MilestonePhase.SEARCH:
            self.phase = MilestonePhase.DETAIL
        elif self.phase is MilestonePhase.DETAIL:
            self.phase = MilestonePhase.OPTIONS
        elif self.phase is MilestonePhase.OPTIONS:
            self.phase = MilestonePhase.PURCHASE
        elif self.phase is MilestonePhase.PURCHASE:
            self.phase = MilestonePhase.COMPLETE

    def _log_session_init(self) -> None:
        """记录会话初始化信息"""
        if not self.log_file:
            return

        init_text = f"""{'='*80}
SESSION INITIALIZED: {self.session_id}
{'='*80}
Goal Instruction:
{self.goal.get('instruction_text', 'N/A')}

Goal Attributes:
{chr(10).join(f'- {attr}' for attr in self.goal.get('attributes', []))}

Goal Options:
{chr(10).join(f'- {opt}' for opt in sorted(self.goal_option_values))}

Price Upper Limit: {self.goal.get('price_upper', 'N/A')}
{'='*80}

"""
        with open(self.log_file, 'w') as f:
            f.write(init_text)

    def _log_step(
        self,
        action: str,
        state: str,
        next_state: str,
        result: MilestoneResult,
    ) -> None:
        """记录调试日志到文件和控制台（任务执行进度报告格式）"""
        self.step_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 添加到历史记录
        status_symbol = "✓ ACHIEVED" if result.achieved else "✗ Not Achieved"
        self.action_history.append({
            'action': action,
            'phase': result.phase.name,
            'status': status_symbol,
            'message': result.message
        })

        # 控制台输出（简洁）
        action_preview = action[:60] + "..." if len(action) > 60 else action
        print(f"[Step {self.step_count}] Action: {action_preview}")
        print(f"  Phase: {result.phase.name} | Status: {status_symbol}")
        print(f"  Message: {result.message}")

        if result.metadata:
            for key, value in result.metadata.items():
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value)
                else:
                    value_str = str(value)
                print(f"  {key}: {value_str}")
        print()  # 空行分隔

        # 文件输出（完整的任务执行进度报告）
        if self.log_file:
            # 格式化任务描述
            task_text = self.goal.get('instruction_text', 'N/A')

            # 格式化当前状态
            current_state_formatted = self._format_state(state)

            # 格式化历史动作
            if len(self.action_history) > 1:
                history_lines = []
                for i, hist in enumerate(self.action_history[:-1], 1):  # 排除当前动作
                    history_lines.append(
                        f"{i}. {hist['action']} → {hist['phase']}: {hist['status']}"
                    )
                action_history_text = "\n".join(history_lines)
            else:
                action_history_text = "(none)"

            # 格式化当前动作
            current_action_text = f"{self.step_count}. {action}"

            # 格式化里程碑检查结果
            milestone_status = "✓ ACHIEVED" if result.achieved else "✗ Not Achieved"
            metadata_text = ""
            if result.metadata:
                metadata_lines = []
                for key, value in result.metadata.items():
                    if isinstance(value, list):
                        value_str = ", ".join(str(v) for v in value)
                    else:
                        value_str = str(value)
                    metadata_lines.append(f"{key}: {value_str}")
                metadata_text = f"Metadata: {'; '.join(metadata_lines)}\n"

            log_entry = f"""{'='*80}
[Step {self.step_count}] {timestamp}
{'='*80}
Task:
{task_text}

Current State:
{current_state_formatted}

Action History:
{action_history_text}

Current Action:
{current_action_text}

Milestone Check:
Phase: {result.phase.name}
Status: {milestone_status}
Message: {result.message}
{metadata_text}{'='*80}

"""
            with open(self.log_file, 'a') as f:
                f.write(log_entry)

    def _format_state(self, state: str, max_line_length: int = 78) -> str:
        """格式化状态文本，添加适当的换行和缩进"""
        if not state:
            return "(empty)"

        # 按 [SEP] 分割
        parts = state.split('[SEP]')
        formatted_parts = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # 如果部分太长，进行换行
            if len(part) <= max_line_length:
                formatted_parts.append(part)
            else:
                # 简单的换行处理
                words = part.split()
                lines = []
                current_line = []
                current_length = 0

                for word in words:
                    if current_length + len(word) + 1 <= max_line_length:
                        current_line.append(word)
                        current_length += len(word) + 1
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)

                if current_line:
                    lines.append(' '.join(current_line))

                formatted_parts.append('\n'.join(lines))

        return '\n[SEP]\n'.join(formatted_parts)

    def _product_can_meet_goal(self, asin: str) -> bool:
        product = self.product_lookup.get(asin.upper())
        if product is None:
            return False

        r_att, num_attr_matches = get_attribute_reward(product, self.goal)
        if num_attr_matches < self.goal_attr_count:
            return False

        if self.goal_option_values:
            product_option_values = self._flatten_product_options(product)
            if not self.goal_option_values.issubset(product_option_values):
                return False

        return self._price_is_valid(asin)

    def _product_fully_satisfies_goal(self, asin: str) -> bool:
        """Final-stage check using selected options."""
        product = self.product_lookup.get(asin.upper())
        if product is None:
            return False

        r_att, num_attr_matches = get_attribute_reward(product, self.goal)
        if num_attr_matches < self.goal_attr_count:
            return False

        if self.goal_option_values and not self.goal_option_values.issubset(
            self.matched_option_values
        ):
            return False

        return self._price_is_valid(asin)

    def _price_is_valid(self, asin: str) -> bool:
        price_upper = self.goal.get("price_upper", 0)
        if price_upper <= 0:
            return True
        price = self.price_lookup.get(asin.upper(), float("inf"))
        return price <= price_upper

    def _flatten_product_options(self, product: Dict[str, Any]) -> Set[str]:
        option_values: Set[str] = set()
        options = product.get("options", {})
        for values in options.values():
            for v in values:
                option_values.add(self._normalize_text(v))
        return option_values

    def _extract_asins_from_state(self, state: str) -> List[str]:
        return [match.group(1).upper() for match in self.ASIN_PATTERN.finditer(state)]

    def _normalize_goal_options(self, goal_options: Any) -> Set[str]:
        if goal_options is None:
            return set()
        if isinstance(goal_options, dict):
            values: Iterable[str] = goal_options.values()
        else:
            values = goal_options
        return {self._normalize_text(v) for v in values if v}

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())


# -----------------------------------------------------------------------------
# Convenience utilities for manual testing
# -----------------------------------------------------------------------------
def load_default_catalog(
    num_products: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """
    Load WebShop products using default JSON files.

    Returns:
        product_lookup: mapping from ASIN -> product dict
        price_lookup: mapping from ASIN -> synthetic price sampled by loader
    """
    all_products, product_lookup, product_prices, _ = load_products(
        filepath=DEFAULT_FILE_PATH,
        attrpath=DEFAULT_ATTR_PATH,
        num_products=num_products,
        human_goals=True,
    )
    # The helper already returns product_lookup, but we re-cast keys to str
    return product_lookup, product_prices


def load_human_goals(
    path: Path = Path(
        "agent_system/environments/env_package/webshop/webshop/data/items_human_ins.json"
    )
) -> Dict[str, List[Dict[str, Any]]]:
    """Utility to load the raw human goal descriptions."""
    with path.open("r") as f:
        return json.load(f)


if __name__ == "__main__":
    product_lookup, price_lookup = load_default_catalog()
    human_goals = load_human_goals()
    # Grab the first available goal entry for demonstration purposes.
    first_asin, goal_entries = next(iter(human_goals.items()))
    raw_goal = goal_entries[0]
    goal = {
        "asin": first_asin,
        "instruction_text": raw_goal["instruction"],
        "attributes": raw_goal["instruction_attributes"],
        "goal_options": raw_goal["instruction_options"],
        "price_upper": 1e6,  # Placeholder bound for manual testing
    }
    detector = MilestoneDetector(goal, product_lookup, price_lookup)
    print(
        "Initialized detector for instruction:",
        goal["instruction_text"],
        "\nGoal options:",
        goal["goal_options"],
    )
    print(
        "Use MilestoneDetector.process(action, state, next_state, info) "
        "to feed environment transitions."
    )
