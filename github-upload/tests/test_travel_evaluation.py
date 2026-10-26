import os

import pytest

from travel_evaluation import (
    DEFAULT_DATASET_PATH,
    load_evaluation_cases,
    run_evaluation,
)


def test_evaluation_dataset_is_valid_and_unique():
    cases = load_evaluation_cases(DEFAULT_DATASET_PATH)
    ids = [case["id"] for case in cases]

    assert len(cases) >= 8
    assert len(ids) == len(set(ids))


def test_offline_evaluation_dataset_passes():
    results = run_evaluation()
    failures = {
        result.case_id: [violation.message for violation in result.violations]
        for result in results
        if not result.passed
    }

    assert not failures


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TRAVEL_EVAL") != "1",
    reason="设置 RUN_LIVE_TRAVEL_EVAL=1 后才调用本地 Ollama",
)
def test_live_ollama_evaluation_dataset_passes():
    results = run_evaluation(live=True)
    failures = {
        result.case_id: [violation.message for violation in result.violations]
        for result in results
        if not result.passed
    }

    assert not failures
