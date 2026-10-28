import json

import pytest

from travel_assistant import (
    ItineraryStop,
    Place,
    TravelAssistant,
    TravelPlan,
    TravelPreferences,
    create_default_places,
)
from travel_evaluation import StaticPlaceCatalog, evaluate_travel_constraints


def make_preferences(**overrides):
    values = {
        "city": "杭州",
        "mood": "想安静",
        "activity_preference": "室内为主",
        "budget": "经济",
        "duration_hours": 4,
        "companions": "独自出行",
        "departure": "市中心",
    }
    values.update(overrides)
    return TravelPreferences(**values)


def make_stop(place_name: str, duration: float = 2) -> ItineraryStop:
    return ItineraryStop("上午", place_name, "参观", duration, "步行", "免费")


def test_catalog_filters_indoor_budget_city_and_duration():
    catalog = StaticPlaceCatalog(create_default_places())
    candidates = catalog.search(make_preferences(duration_hours=2))

    assert candidates
    assert len(candidates) <= 3
    assert all(place.city == "杭州" for place in candidates)
    assert all(place.environment == "室内" for place in candidates)
    assert all(place.budget_level in {"免费", "经济"} for place in candidates)
    assert all(place.recommended_hours <= 2 for place in candidates)


def test_evaluator_reports_outdoor_and_over_budget_places():
    preferences = make_preferences(budget="免费")
    candidates = [
        Place("违规地点", "杭州", "户外", "宽松", 2, (), "测试数据")
    ]
    plan = TravelPlan(
        "测试",
        (make_stop("违规地点"),),
        "测试预算",
        (),
    )

    codes = {
        violation.code
        for violation in evaluate_travel_constraints(preferences, candidates, plan)
    }

    assert "outdoor_for_indoor" in codes
    assert "over_budget" in codes


@pytest.mark.parametrize(
    ("place", "error"),
    [
        (Place("户外地点", "杭州", "户外", "免费", 2, (), ""), "违反室内偏好"),
        (Place("昂贵地点", "杭州", "室内", "宽松", 2, (), ""), "超过用户预算"),
    ],
)
def test_plan_parser_rejects_constraint_violations(place, error):
    payload = {
        "summary": "测试",
        "itinerary": [
            {
                "period": "上午",
                "place_name": place.name,
                "activity": "参观",
                "duration_hours": 2,
                "transport": "步行",
                "estimated_cost": "待核实",
            }
        ],
        "budget_summary": "测试",
        "reminders": [],
    }

    with pytest.raises(ValueError, match=error):
        TravelAssistant._parse_plan(
            json.dumps(payload, ensure_ascii=False),
            make_preferences(budget="免费"),
            [place],
        )


def test_plan_parser_rejects_total_duration_over_limit():
    place = Place("室内地点", "杭州", "室内", "免费", 2, (), "")
    payload = {
        "summary": "测试",
        "itinerary": [
            {
                "period": "上午",
                "place_name": place.name,
                "activity": "参观",
                "duration_hours": 5,
                "transport": "步行",
                "estimated_cost": "免费",
            }
        ],
        "budget_summary": "免费",
        "reminders": [],
    }

    with pytest.raises(ValueError, match="总行程时长超过"):
        TravelAssistant._parse_plan(
            json.dumps(payload, ensure_ascii=False),
            make_preferences(),
            [place],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("activity_preference", "随便"), ("budget", "无限")],
)
def test_preferences_reject_unknown_constraint_values(field, value):
    preferences = make_preferences(**{field: value})
    with pytest.raises(ValueError):
        preferences.validate()


class SequencedLLM:
    """按顺序返回预设结果，用于模拟小模型连续输出库外地点。"""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        return self.answers.pop(0)


def invalid_plan(place_name: str) -> str:
    return json.dumps(
        {
            "summary": "测试",
            "itinerary": [
                {
                    "period": "上午",
                    "place_name": place_name,
                    "activity": "参观",
                    "duration_hours": 1,
                    "transport": "步行",
                    "estimated_cost": "经济",
                }
            ],
            "budget_summary": "经济",
            "reminders": [],
        },
        ensure_ascii=False,
    )


def test_plan_request_falls_back_to_trusted_candidates_after_two_violations():
    candidate = Place(
        "安静书店",
        "杭州",
        "室内",
        "经济",
        2,
        ("想安静", "独自出行"),
        "适合安静阅读",
    )
    llm = SequencedLLM(
        [invalid_plan("艺术画廊"), invalid_plan("湖滨步行")]
    )
    assistant = TravelAssistant(llm)

    plan = assistant._request_plan([], make_preferences(), [candidate])

    assert [stop.place_name for stop in plan.itinerary] == [candidate.name]
    assert sum(stop.duration_hours for stop in plan.itinerary) <= 4
    assert "安静书店" in llm.calls[1][-1]["content"]
    assert "程序使用可信候选生成" in plan.reminders[0]
