"""智能旅行助手的确定性约束检查和可选 LLM 评测。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from travel_assistant import (
    Place,
    PlaceCatalog,
    TRAVEL_BUDGET_RANK,
    TravelAssistant,
    TravelPlan,
    TravelPreferences,
    TravelToolDecision,
    TravelToolRouter,
    create_default_places,
    create_llm,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = PROJECT_DIR / "evals" / "travel_constraints.json"


@dataclass(frozen=True)
class ConstraintViolation:
    """一条可机器判断的约束违规。"""

    code: str
    message: str


@dataclass(frozen=True)
class EvaluationResult:
    """一个评测案例的结果。"""

    case_id: str
    passed: bool
    violations: tuple[ConstraintViolation, ...]
    decision: dict[str, bool]
    candidate_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "violations": [asdict(item) for item in self.violations],
            "decision": self.decision,
            "candidate_names": list(self.candidate_names),
        }


class StaticPlaceCatalog(PlaceCatalog):
    """评测使用的只读地点目录，不接触业务数据库。"""

    def __init__(self, places: tuple[Place, ...]):
        self.places = places

    def search(
        self,
        preferences: TravelPreferences,
        limit: int = 3,
    ) -> list[Place]:
        return self._filter_places(self.places, preferences, limit)


def load_evaluation_cases(path: str | Path = DEFAULT_DATASET_PATH) -> list[dict[str, Any]]:
    """读取并执行评测集的最低结构校验。"""
    dataset_path = Path(path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("评测集必须是非空 JSON 数组")

    required = {
        "id",
        "question",
        "preferences",
        "expected_tools",
        "expect_itinerary",
    }
    seen_ids: set[str] = set()
    for case in payload:
        if not isinstance(case, dict) or not required.issubset(case):
            raise ValueError("评测案例缺少必要字段")
        case_id = str(case["id"])
        if case_id in seen_ids:
            raise ValueError(f"评测案例 ID 重复：{case_id}")
        seen_ids.add(case_id)
        TravelPreferences(**case["preferences"]).validate()
        expected_tools = case["expected_tools"]
        if set(expected_tools) != {"place_search", "weather", "routes", "knowledge"}:
            raise ValueError(f"案例 {case_id} 的 expected_tools 字段不完整")
        if any(not isinstance(value, bool) for value in expected_tools.values()):
            raise ValueError(f"案例 {case_id} 的工具期望必须是布尔值")
    return payload

##把路由器返回的 TravelToolDecision 对象（带 use_ 前缀），拍平成一个普通的字典 {"place_search": True, "weather": False, ...}
def decision_flags(decision: TravelToolDecision) -> dict[str, bool]:
    return {
        "place_search": decision.use_place_search,
        "weather": decision.use_weather,
        "routes": decision.use_routes,
        "knowledge": decision.use_knowledge,
    }


def evaluate_tool_decision(
    decision: TravelToolDecision,
    expected: dict[str, bool],
) -> list[ConstraintViolation]:
    """比较实际和期望的工具开关。"""
    actual = decision_flags(decision)
    violations = []
    for tool_name, expected_value in expected.items():
        if actual[tool_name] != expected_value:
            violations.append(
                ConstraintViolation(
                    "wrong_tool_decision",
                    f"工具 {tool_name} 期望为 {expected_value}，实际为 {actual[tool_name]}",
                )
            )
    return violations


def evaluate_travel_constraints(
    preferences: TravelPreferences,
    candidates: list[Place],
    plan: TravelPlan | None = None,
    expect_itinerary: bool | None = None,
) -> list[ConstraintViolation]:
    """检查地点候选和最终行程是否违反硬约束。"""
    violations: list[ConstraintViolation] = []
    budget_limit = TRAVEL_BUDGET_RANK[preferences.budget]

    def check_place(place: Place, location: str) -> None:
        if place.city.casefold() != preferences.city.strip().casefold():
            violations.append(
                ConstraintViolation("wrong_city", f"{location}包含其他城市地点：{place.name}")
            )
        place_budget = TRAVEL_BUDGET_RANK.get(place.budget_level)
        if place_budget is None or place_budget > budget_limit:
            violations.append(
                ConstraintViolation("over_budget", f"{location}包含超预算地点：{place.name}")
            )
        if place.recommended_hours > preferences.duration_hours:
            violations.append(
                ConstraintViolation("place_too_long", f"{location}地点建议时长超限：{place.name}")
            )
        if (
            preferences.activity_preference == "室内为主"
            and place.environment != "室内"
        ):
            violations.append(
                ConstraintViolation("outdoor_for_indoor", f"{location}违反室内偏好：{place.name}")
            )
        if (
            preferences.activity_preference == "户外为主"
            and place.environment != "户外"
        ):
            violations.append(
                ConstraintViolation("indoor_for_outdoor", f"{location}违反户外偏好：{place.name}")
            )

    if len(candidates) > 3:
        violations.append(ConstraintViolation("too_many_candidates", "候选地点超过 3 个"))
    for candidate in candidates:
        check_place(candidate, "候选列表")

    if plan is None:
        return violations

    candidate_by_name = {place.name: place for place in candidates}
    total_hours = 0.0
    for stop in plan.itinerary:
        total_hours += stop.duration_hours
        place = candidate_by_name.get(stop.place_name)
        if place is None:
            violations.append(
                ConstraintViolation(
                    "unknown_place",
                    f"行程使用候选列表之外的地点：{stop.place_name}",
                )
            )
            continue
        check_place(place, "最终行程")
    if total_hours > preferences.duration_hours:
        violations.append(
            ConstraintViolation(
                "duration_exceeded",
                f"行程总时长 {total_hours:g} 小时超过可用时间",
            )
        )
    if expect_itinerary is True and candidates and not plan.itinerary:
        violations.append(ConstraintViolation("missing_itinerary", "存在候选地点但模型未生成行程"))
    if expect_itinerary is False and plan.itinerary:
        violations.append(ConstraintViolation("unexpected_itinerary", "本案例不应生成地点行程"))
    return violations


def evaluate_case(
    case: dict[str, Any],
    catalog: StaticPlaceCatalog,
    live: bool,
    llm: Any | None = None,
) -> EvaluationResult:
    """执行一个离线规则案例或真实本地模型案例。"""
    preferences = TravelPreferences(**case["preferences"])
    if live:
        if llm is None:
            raise ValueError("真实评测需要 LLM")
        router = TravelToolRouter(llm)
        decision = router.decide(preferences, case["question"], [])
    else:
        decision = TravelToolRouter._fallback(case["question"])

    violations = evaluate_tool_decision(decision, case["expected_tools"])
    candidates = catalog.search(preferences) if decision.use_place_search else []
    plan = None

    if live:
        assistant = TravelAssistant(
            llm,
            place_catalog=catalog,
            tool_router=router,
        )
        try:
            messages = assistant.build_messages(
                preferences,
                case["question"],
                candidates=candidates,
                tool_decision=decision,
            )
            plan = assistant._request_plan(messages, preferences, candidates)
        except Exception as exc:
            violations.append(ConstraintViolation("model_error", str(exc)))

    violations.extend(
        evaluate_travel_constraints(
            preferences,
            candidates,
            plan,
            case["expect_itinerary"] if live and plan is not None else None,
        )
    )
    return EvaluationResult(
        case_id=case["id"],
        passed=not violations,
        violations=tuple(violations),
        decision=decision_flags(decision),
        candidate_names=tuple(place.name for place in candidates),
    )


def run_evaluation(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    live: bool = False,
) -> list[EvaluationResult]:
    """运行完整评测集；live=True 时调用本地 Ollama。"""
    cases = load_evaluation_cases(dataset_path)
    catalog = StaticPlaceCatalog(create_default_places())
    llm = create_llm() if live else None
    return [evaluate_case(case, catalog, live, llm) for case in cases]


def main() -> int:
    parser = argparse.ArgumentParser(description="智能旅行助手约束评测")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--live", action="store_true", help="调用本地 Ollama 做端到端评测")
    parser.add_argument("--output", help="可选的 JSON 评测报告路径")
    args = parser.parse_args()

    results = run_evaluation(args.dataset, live=args.live)
    for result in results:
        status = "通过" if result.passed else "失败"
        print(f"[{status}] {result.case_id}")
        for violation in result.violations:
            print(f"  - {violation.code}: {violation.message}")

    passed_count = sum(result.passed for result in results)
    print(f"评测完成：{passed_count}/{len(results)} 通过")
    if args.output:
        report = {
            "mode": "live" if args.live else "offline",
            "passed": passed_count,
            "total": len(results),
            "results": [result.to_dict() for result in results],
        }
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
