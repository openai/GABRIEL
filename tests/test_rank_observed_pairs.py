import asyncio
import math

import numpy as np
import pandas as pd
import pytest

import gabriel
from gabriel.tasks.rank import Rank, RankConfig


@pytest.fixture
def ranker(tmp_path):
    return Rank(RankConfig(attributes={"clarity": ""}, save_dir=str(tmp_path)))


def test_unrelated_comparisons_leave_raw_contrast_and_error_unchanged(ranker):
    outcomes = [("A", "B")] * 9 + [("B", "A")] * 3
    scores, counts, probabilities = ranker._fit_bt(
        ["A", "B", "C", "D", "E"],
        outcomes + [("C", "D")],
        pseudo=2,
        max_iter=1000,
        tol=1e-6,
        return_info=True,
    )
    alone, alone_counts, alone_probabilities = ranker._fit_bt(
        ["A", "B"], outcomes, pseudo=2, max_iter=1000, tol=1e-6, return_info=True
    )
    assert scores["A"] - scores["B"] == pytest.approx(math.log(10 / 4))
    assert [scores["A"], scores["B"]] == pytest.approx(list(alone.values()))
    assert counts[0, 1] == 12
    assert counts[0, 2] == 0
    assert np.all(np.diag(counts) == 0)

    errors = ranker._bt_standard_errors(counts, probabilities, ridge=0, pseudo=2)
    alone_errors = ranker._bt_standard_errors(
        alone_counts, alone_probabilities, ridge=0, pseudo=2
    )
    probability = 10 / 14
    expected_error = math.sqrt(12 / (4 * 14**2 * probability * (1 - probability)))
    assert errors[:2] == pytest.approx([expected_error, expected_error])
    assert errors[:2] == pytest.approx(alone_errors)
    assert np.isnan(errors[4])


def test_connected_three_item_fit_satisfies_score_equations(ranker):
    outcomes = [("A", "B")] * 5 + [("B", "A")] + [("B", "C")] * 4
    outcomes += [("C", "B")] * 2 + [("A", "C")] * 2 + [("C", "A")] * 3
    scores, counts, probabilities = ranker._fit_bt(
        ["A", "B", "C"], outcomes, pseudo=2, max_iter=1000, tol=1e-8, return_info=True
    )
    expected_wins = ((counts + 2 * (counts > 0)) * probabilities).sum(axis=1)
    assert expected_wins == pytest.approx([9, 7, 7], abs=1e-6)
    assert sum(scores.values()) == pytest.approx(0)


def test_zero_smoothing_requires_a_finite_fit(ranker):
    with pytest.raises(ValueError, match="separated comparison component"):
        ranker._fit_bt(["A", "B"], [("A", "B")], pseudo=0, max_iter=100, tol=1e-6)
    scores = ranker._fit_bt(
        ["A", "B"],
        [("A", "B"), ("A", "B"), ("B", "A")],
        pseudo=0,
        max_iter=100,
        tol=1e-6,
    )
    assert scores["A"] - scores["B"] == pytest.approx(math.log(2))


def test_rank_reports_the_analytic_contrast_from_one_recorded_judgment(tmp_path):
    async def comparisons(prompts, identifiers, **kwargs):
        assert len(prompts) == len(identifiers) == 1
        return pd.DataFrame(
            {"Identifier": identifiers, "Response": ['{"clarity": "circle"}']}
        )

    result = asyncio.run(
        gabriel.rank(
            pd.DataFrame({"text": ["first", "second"]}),
            "text",
            attributes={"clarity": "clarity of the passage"},
            save_dir=str(tmp_path),
            n_rounds=1,
            matches_per_round=1,
            power_matching=False,
            initial_rating_pass=False,
            learning_rate=2,
            return_raw_scores=True,
            get_all_responses_fn=comparisons,
        )
    )
    assert result["clarity_raw"].max() - result["clarity_raw"].min() == pytest.approx(
        math.log(2)
    )
    assert np.isfinite(result["clarity_se"]).all()
