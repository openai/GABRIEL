import asyncio
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from gabriel.tasks.rank import Rank, RankConfig


def _rank(tmp_path, *, insufficient_signal_policy="tie"):
    return Rank(
        RankConfig(
            attributes={"quality": ""},
            save_dir=str(tmp_path),
            initial_rating_pass=False,
            insufficient_signal_policy=insufficient_signal_policy,
        )
    )


def _fit(rank, item_ids, outcomes, *, pseudo=0.1):
    return rank._fit_bt(
        item_ids=list(item_ids),
        outcomes=outcomes,
        pseudo=pseudo,
        max_iter=20_000,
        tol=1e-12,
        return_info=True,
    )


def _standard_errors(
    rank, item_ids, scores, n_ij, p_ij, *, regularization_strength=0.1
):
    return rank._bt_standard_errors(
        s=np.array([scores[item] for item in item_ids]),
        n_ij=n_ij,
        p_ij=p_ij,
        rcond=rank._SE_EIGEN_TOL,
        regularization_strength=regularization_strength,
    )


def test_outcome_decoder_uses_unit_weight_draws_and_bounded_labels(tmp_path):
    rank = _rank(tmp_path)

    assert rank._decode_pairwise_outcome("a", "b", "draw") == (
        "draw",
        [("a", "b", 0.5), ("b", "a", 0.5)],
    )
    assert rank._decode_pairwise_outcome("a", "b", {"Winner": "circle"}) == (
        "circle",
        [("a", "b", 1.0)],
    )
    assert rank._decode_pairwise_outcome("a", "b", "square wins") == (
        "square",
        [("b", "a", 1.0)],
    )
    assert rank._decode_pairwise_outcome("a", "b", "cannot determine") == (
        "invalid",
        [],
    )
    assert rank._decode_pairwise_outcome("a", "b", "both lack evidence") == (
        "invalid",
        [],
    )
    for ambiguous in (
        "a tie",
        "a draw",
        "A: insufficient signal",
        "b: cannot determine",
        "c: cannot determine",
        "circle is worse",
        "square loses",
        "draw but square is stronger",
        "left",
        "right",
    ):
        assert rank._decode_pairwise_outcome("a", "b", ambiguous) == (
            "invalid",
            [],
        )


def test_insufficient_signal_policy_is_explicit(tmp_path):
    tie_rank = _rank(tmp_path / "tie", insufficient_signal_policy="tie")
    abstain_rank = _rank(tmp_path / "abstain", insufficient_signal_policy="abstain")

    assert tie_rank._decode_pairwise_outcome(
        "a", "b", "insufficient signal"
    ) == (
        "insufficient_signal",
        [("a", "b", 0.5), ("b", "a", 0.5)],
    )
    assert abstain_rank._decode_pairwise_outcome(
        "a", "b", "insufficient signal"
    ) == ("insufficient_signal", [])

    with pytest.raises(ValueError, match="insufficient_signal_policy"):
        RankConfig(
            attributes={"quality": ""},
            insufficient_signal_policy="guess",
        )
    for invalid_rate in (True, "0.1", np.inf, -0.1):
        with pytest.raises(ValueError, match="learning_rate"):
            RankConfig(
                attributes={"quality": ""}, learning_rate=invalid_rate
            )
    normalized = RankConfig(
        attributes={"quality": ""}, learning_rate=np.float64(0.25)
    )
    assert normalized.learning_rate == 0.25
    for invalid_scale in (True, "1", np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="primer_scale"):
            RankConfig(attributes={"quality": ""}, primer_scale=invalid_scale)
    with pytest.raises(ValueError, match="primer_center"):
        RankConfig(attributes={"quality": ""}, primer_center=1)
    with pytest.raises(ValueError, match="judge_version"):
        RankConfig(attributes={"quality": ""}, judge_version="  ")

    zero_primer = RankConfig(
        attributes={"quality": ""}, primer_scale=0, primer_center=True
    )
    ratings = {"a": {"quality": 0.0}, "b": {"quality": 0.0}}
    Rank(zero_primer)._apply_primer(
        ratings,
        {"a": {"quality": 10.0}, "b": {"quality": 0.0}},
        ["quality"],
    )
    assert ratings == {"a": {"quality": 0.0}, "b": {"quality": 0.0}}
    for suffix in ("raw", "se", "component"):
        with pytest.raises(ValueError, match="namespaces overlap"):
            RankConfig(attributes={"foo": "", f"foo_{suffix}": ""})
    for reserved in ("identifier", "overall_rank", "exit_stage", "stage1_quality"):
        with pytest.raises(ValueError, match="Recursive Rank attribute names"):
            RankConfig(attributes={reserved: ""}, recursive=True)
    for reserved_key in ("attributes", "save_dir", "file_name"):
        with pytest.raises(ValueError, match="Rank-owned"):
            RankConfig(
                attributes={"quality": ""},
                rate_kwargs={reserved_key: "not-allowed"},
            )
    mutable_config = RankConfig(attributes={"quality": ""})
    mutable_config.rate_kwargs["save_dir"] = str(tmp_path / "outside")
    with pytest.raises(ValueError, match="Rank-owned"):
        Rank(mutable_config)._split_rate_kwargs()
    assert type(normalized.learning_rate) is float
    for field_name in ("n_rounds", "matches_per_round"):
        with pytest.raises(ValueError, match=field_name):
            RankConfig(attributes={"quality": ""}, **{field_name: 0})
        with pytest.raises(ValueError, match=field_name):
            RankConfig(attributes={"quality": ""}, **{field_name: True})
    for empty_attributes in ({}, []):
        with pytest.raises(ValueError, match="at least one attribute"):
            RankConfig(attributes=empty_attributes)
    for invalid_attributes in ([""], [1], ["Quality", " quality "]):
        with pytest.raises(ValueError, match="attribute names"):
            RankConfig(attributes=invalid_attributes)
    for invalid_fraction in (True, 0, 1, -0.1, np.inf):
        with pytest.raises(ValueError, match="recursive_fraction"):
                RankConfig(
                    attributes={"quality": ""},
                    recursive_fraction=invalid_fraction,
                )
    for field_name in (
        "recursive_min_remaining",
        "recursive_final_round_multiplier",
    ):
        for invalid_value in (True, 0, 1.5):
            with pytest.raises(ValueError, match=field_name):
                RankConfig(
                    attributes={"quality": ""},
                    **{field_name: invalid_value},
                )


def test_persisted_rank_attributes_cannot_bypass_namespace_validation(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "attributes.json").write_text(
        json.dumps({"foo": "", "foo_raw": ""})
    )
    rank = Rank(
        RankConfig(
            attributes={"quality": ""},
            save_dir=str(tmp_path),
            initial_rating_pass=False,
        )
    )

    with pytest.raises(ValueError, match="namespaces overlap"):
        asyncio.run(rank.run(pd.DataFrame({"text": ["A", "B"]}), "text"))


@pytest.mark.parametrize("reserved_column", ["overall_rank", "exit_stage", "stage2_x"])
def test_recursive_rank_rejects_reserved_input_columns_before_judging(
    tmp_path, reserved_column
):
    rank = Rank(
        RankConfig(
            attributes={"quality": ""},
            save_dir=str(tmp_path),
            recursive=True,
            initial_rating_pass=False,
        )
    )
    data = pd.DataFrame({"text": ["A", "B"], reserved_column: [0, 1]})

    with pytest.raises(ValueError, match="input columns cannot use internal"):
        asyncio.run(rank.run(data, "text"))


def test_power_matching_flag_selects_random_scheduler_and_caps_degree(
    monkeypatch, tmp_path
):
    rank = Rank(
        RankConfig(
            attributes={"quality": ""},
            save_dir=str(tmp_path),
            initial_rating_pass=False,
            power_matching=False,
            matches_per_round=99,
        )
    )
    observed = {}

    def fake_random(item_ids, texts_by_id, mpr):
        observed["args"] = (item_ids, texts_by_id, mpr)
        return [(('a', 'A'), ('b', 'B'))]

    def fail_info_gain(*args, **kwargs):
        raise AssertionError("power_matching=False must not use the heuristic")

    monkeypatch.setattr(rank, "_pairs_random", fake_random)
    monkeypatch.setattr(rank, "_pairs_info_gain", fail_info_gain)
    result = rank._generate_pairs(
        ["a", "b", "c"],
        {"a": "A", "b": "B", "c": "C"},
        current_ratings={"a": 0.0, "b": 0.0, "c": 0.0},
        se_agg={"a": 1.0, "b": 1.0, "c": 1.0},
    )

    assert result == [(('a', 'A'), ('b', 'B'))]
    assert observed["args"][2] == 2


def test_initial_rate_seed_uses_media_identifier_semantics(tmp_path):
    rank_module = importlib.import_module("gabriel.tasks.rank")
    payloads = [["a.png", "b.png"], ["c.png"]]
    item_ids = [
        rank_module._hash_text_identifier(payload, strict=False, bits=64)
        for payload in payloads
    ]
    rank = Rank(
        RankConfig(
            attributes={"quality": ""},
            modality="image",
            save_dir=str(tmp_path),
        )
    )

    seeds = rank._seed_ratings_from_rate(
        pd.DataFrame({"image": payloads, "quality": [20.0, 80.0]}),
        id_column=None,
        text_column="image",
        item_ids=item_ids,
        attr_keys=["quality"],
        identifier_hash_bits=64,
    )

    assert set(seeds) == set(item_ids)
    assert seeds[item_ids[0]]["quality"] == pytest.approx(-30.0)
    assert seeds[item_ids[1]]["quality"] == pytest.approx(30.0)


def test_regularization_is_derived_from_a_coherent_win_matrix(tmp_path):
    rank = _rank(tmp_path)
    observed, fitted = rank._build_bt_win_matrices(
        ["a", "b", "c"],
        [("a", "b", 1.0), ("b", "a", 0.5)],
        pseudo=0.1,
    )

    np.testing.assert_allclose(
        observed,
        [[0.0, 1.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        fitted,
        [[0.0, 1.05, 0.0], [0.55, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(np.diag(fitted), 0.0)
    assert fitted[0, 2] == fitted[2, 0] == 0.0
    np.testing.assert_allclose(fitted + fitted.T, (fitted + fitted.T).T)
    np.testing.assert_allclose(fitted.sum(axis=1), [1.05, 0.55, 0.0])
    assert fitted.sum() == pytest.approx(
        np.triu(fitted + fitted.T, k=1).sum()
    )


def test_two_item_fit_matches_regularized_closed_form(tmp_path):
    rank = _rank(tmp_path)
    outcomes = [("a", "b")] * 9 + [("b", "a")]

    scores, n_ij, p_ij = _fit(rank, ["a", "b"], outcomes)

    expected_gap = np.log((9.0 + 0.05) / (1.0 + 0.05))
    assert scores["a"] - scores["b"] == pytest.approx(expected_gap, abs=1e-10)
    np.testing.assert_allclose(n_ij, [[0.0, 10.0], [10.0, 0.0]])
    assert p_ij[0, 1] / p_ij[1, 0] == pytest.approx(np.exp(expected_gap))


@pytest.mark.parametrize("n_items", [2, 5, 10, 50, 100])
def test_balanced_extensions_do_not_distort_existing_contrasts(tmp_path, n_items):
    rank = _rank(tmp_path)
    item_ids = ["a", "b"] + [f"u{i}" for i in range(n_items - 2)]
    outcomes = [("a", "b")] * 9 + [("b", "a")]
    for item in item_ids[2:]:
        outcomes.extend([(item, "b")] * 5)
        outcomes.extend([("b", item)] * 5)

    scores, _, _ = _fit(rank, item_ids, outcomes)

    expected_gap = np.log((9.0 + 0.05) / (1.0 + 0.05))
    assert scores["a"] - scores["b"] == pytest.approx(expected_gap, abs=1e-8)
    for item in item_ids[2:]:
        assert scores[item] - scores["b"] == pytest.approx(0.0, abs=1e-8)


def test_fit_is_permutation_equivariant(tmp_path):
    rank = _rank(tmp_path)
    outcomes = [
        ("a", "b", 3.0),
        ("b", "a", 1.0),
        ("b", "c", 2.0),
        ("c", "b", 1.0),
        ("c", "a", 1.5),
        ("a", "c", 0.5),
    ]

    reference, _, _ = _fit(rank, ["a", "b", "c"], outcomes)
    permuted, _, _ = _fit(rank, ["c", "a", "b"], outcomes)

    for item in reference:
        assert permuted[item] == pytest.approx(reference[item], abs=1e-10)


def test_reversing_every_outcome_negates_scores(tmp_path):
    rank = _rank(tmp_path)
    outcomes = [
        ("a", "b", 4.0),
        ("b", "a", 1.0),
        ("b", "c", 3.0),
        ("c", "b", 2.0),
        ("c", "a", 1.0),
    ]
    reversed_outcomes = [(loser, winner, weight) for winner, loser, weight in outcomes]

    scores, _, _ = _fit(rank, ["a", "b", "c"], outcomes)
    reversed_scores, _, _ = _fit(
        rank, ["a", "b", "c"], reversed_outcomes
    )

    for item in scores:
        assert reversed_scores[item] == pytest.approx(-scores[item], abs=1e-9)


def test_bt_fit_matches_constrained_likelihood_optimization(tmp_path):
    rank = _rank(tmp_path)
    item_ids = ["a", "b", "c", "d"]
    outcomes = [
        ("a", "b", 4.0),
        ("b", "a", 1.0),
        ("a", "c", 1.0),
        ("c", "a", 3.0),
        ("b", "c", 2.0),
        ("c", "b", 2.0),
        ("b", "d", 3.0),
        ("d", "b", 1.0),
        ("c", "d", 1.0),
        ("d", "c", 2.0),
    ]
    fitted_scores, _, _ = _fit(rank, item_ids, outcomes)
    _, fitted_wins = rank._build_bt_win_matrices(item_ids, outcomes, pseudo=0.1)

    def objective(free_scores):
        scores = np.append(free_scores, -np.sum(free_scores))
        value = 0.0
        for i in range(len(item_ids)):
            for j in range(len(item_ids)):
                if i != j and fitted_wins[i, j] > 0:
                    value += fitted_wins[i, j] * np.logaddexp(
                        0.0, -(scores[i] - scores[j])
                    )
        return value

    result = minimize(
        objective,
        np.zeros(len(item_ids) - 1),
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 10_000},
    )
    assert result.success or np.linalg.norm(result.jac) < 1e-6
    optimizer_scores = np.append(result.x, -np.sum(result.x))

    np.testing.assert_allclose(
        [fitted_scores[item] for item in item_ids], optimizer_scores, atol=2e-6
    )


def test_draw_has_one_comparison_in_fractional_binomial_working_model(tmp_path):
    rank = _rank(tmp_path)
    _, outcomes = rank._decode_pairwise_outcome("a", "b", "draw")
    scores, n_ij, p_ij = _fit(rank, ["a", "b"], outcomes, pseudo=0.0)
    se = _standard_errors(
        rank,
        ["a", "b"],
        scores,
        n_ij,
        p_ij,
        regularization_strength=0.0,
    )

    assert n_ij[0, 1] == n_ij[1, 0] == pytest.approx(1.0)
    np.testing.assert_allclose(se, [1.0, 1.0], atol=1e-10)

    doubled_scores, doubled_n, doubled_p = _fit(
        rank, ["a", "b"], outcomes + outcomes, pseudo=0.0
    )
    doubled_se = _standard_errors(
        rank,
        ["a", "b"],
        doubled_scores,
        doubled_n,
        doubled_p,
        regularization_strength=0.0,
    )
    np.testing.assert_allclose(doubled_se, se / np.sqrt(2.0), atol=1e-10)


def test_standard_errors_use_penalized_estimator_sandwich(tmp_path):
    rank = _rank(tmp_path)
    outcomes = [("a", "b", 5.0), ("b", "a", 5.0)]

    low_scores, low_n, low_p = _fit(rank, ["a", "b"], outcomes, pseudo=0.01)
    high_scores, high_n, high_p = _fit(rank, ["a", "b"], outcomes, pseudo=10.0)
    low_se = _standard_errors(
        rank,
        ["a", "b"],
        low_scores,
        low_n,
        low_p,
        regularization_strength=0.01,
    )
    high_se = _standard_errors(
        rank,
        ["a", "b"],
        high_scores,
        high_n,
        high_p,
        regularization_strength=10.0,
    )

    np.testing.assert_allclose(low_n, high_n)
    np.testing.assert_allclose(low_se, np.sqrt(10.0) / 10.01, atol=1e-12)
    np.testing.assert_allclose(high_se, np.sqrt(10.0) / 20.0, atol=1e-12)
    assert np.all(high_se < low_se)


def test_standard_errors_remain_identified_after_sigmoid_saturation(tmp_path):
    rank = _rank(tmp_path)
    pseudo = 1e-18
    scores, n_ij, p_ij = _fit(
        rank, ["a", "b"], [("a", "b")], pseudo=pseudo
    )
    se = _standard_errors(
        rank,
        ["a", "b"],
        scores,
        n_ij,
        p_ij,
        regularization_strength=pseudo,
    )

    gap = abs(scores["a"] - scores["b"])
    tail = np.exp(-gap)
    variance = tail / (1.0 + tail) ** 2
    expected = np.sqrt(n_ij[0, 1]) / (
        2.0 * (n_ij[0, 1] + pseudo) * np.sqrt(variance)
    )

    assert p_ij[0, 1] == 1.0
    assert np.isfinite(se).all()
    np.testing.assert_allclose(se, [expected, expected], rtol=1e-12)


@pytest.mark.parametrize("pseudo", [1e-18, 1e-100, 1e-300])
@pytest.mark.parametrize("n_items", [3, 5, 20])
def test_tiny_pseudo_chain_has_stable_closed_form_fit_and_se(
    tmp_path, pseudo, n_items
):
    rank = _rank(tmp_path)
    item_ids = [str(index) for index in range(n_items)]
    outcomes = list(zip(item_ids[:-1], item_ids[1:]))

    scores, n_ij, p_ij = rank._fit_bt(
        item_ids,
        outcomes,
        pseudo=pseudo,
        max_iter=1,
        tol=1e-6,
        return_info=True,
    )
    expected_gap = np.log1p(pseudo / 2.0) - np.log(pseudo / 2.0)
    fitted_gaps = np.array(
        [scores[item_ids[i]] - scores[item_ids[i + 1]] for i in range(n_items - 1)]
    )
    np.testing.assert_allclose(fitted_gaps, expected_gap, atol=2e-8, rtol=0)

    se = _standard_errors(
        rank,
        item_ids,
        scores,
        n_ij,
        p_ij,
        regularization_strength=pseudo,
    )
    assert np.isfinite(se).all()


@pytest.mark.parametrize("pseudo", [1e-18, 1e-100, 1e-300])
def test_tiny_pseudo_mixed_curvature_component_converges(tmp_path, pseudo):
    rank = _rank(tmp_path)
    item_ids = ["0", "1", "2"]
    outcomes = [
        ("1", "0", 1.0),
        ("0", "2", 3.0),
        ("2", "0", 1.0),
        ("1", "2", 1.0),
    ]
    scores = rank._fit_bt(
        item_ids,
        outcomes,
        pseudo=pseudo,
        max_iter=1000,
        tol=1e-6,
    )

    assert all(np.isfinite(list(scores.values())))
    assert scores["1"] > scores["0"] > scores["2"]
    assert scores["0"] - scores["2"] == pytest.approx(np.log(3.0), abs=1e-10)


@pytest.mark.parametrize("pseudo", [1e-16, 1e-100, 1e-300])
def test_tiny_pseudo_separated_sccs_preserve_internal_score_gaps(
    tmp_path, pseudo
):
    rank = _rank(tmp_path)
    item_ids = ["0", "1", "2", "3"]
    outcomes = [
        ("0", "1", 1.0),
        ("0", "2", 1.0),
        ("0", "3", 3.0),
        ("3", "0", 1.0),
        ("1", "2", 3.0),
        ("2", "1", 1.0),
        ("3", "1", 1.0),
        ("3", "2", 1.0),
    ]
    scores = rank._fit_bt(
        item_ids,
        outcomes,
        pseudo=pseudo,
        max_iter=1000,
        tol=1e-10,
    )
    permuted_scores = rank._fit_bt(
        ["0", "2", "3", "1"],
        outcomes,
        pseudo=pseudo,
        max_iter=1000,
        tol=1e-10,
    )

    assert all(np.isfinite(list(scores.values())))
    assert min(scores["0"], scores["3"]) > max(scores["1"], scores["2"])
    assert scores["0"] - scores["3"] == pytest.approx(np.log(3.0), abs=2e-9)
    assert scores["1"] - scores["2"] == pytest.approx(np.log(3.0), abs=2e-9)

    expected_stratum_mean = 0.5 * (-np.log(pseudo) + np.log(8.0 / 3.0))
    top_mean = 0.5 * (scores["0"] + scores["3"])
    bottom_mean = 0.5 * (scores["1"] + scores["2"])
    assert top_mean == pytest.approx(expected_stratum_mean, abs=2e-9)
    assert bottom_mean == pytest.approx(-expected_stratum_mean, abs=2e-9)
    for item_id in item_ids:
        assert permuted_scores[item_id] == pytest.approx(
            scores[item_id], abs=2e-9
        )


def test_tiny_pseudo_unequal_scc_sizes_converge_without_root_quantization(
    tmp_path,
):
    rank = _rank(tmp_path)
    item_ids = ["0", "1", "2", "3"]
    outcomes = [
        ("0", "1", 3.0),
        ("1", "0", 1.0),
        ("2", "0", 1.0),
        ("3", "0", 1.0),
        ("2", "1", 1.0),
        ("1", "3", 1.0),
        ("2", "3", 1.0),
    ]
    pseudo = 1e-18
    scores = rank._fit_bt(
        item_ids,
        outcomes,
        pseudo=pseudo,
        max_iter=1000,
        tol=1e-6,
    )
    observed_wins, fit_wins = rank._build_bt_win_matrices(
        item_ids, outcomes, pseudo
    )
    fitted = np.array([scores[item_id] for item_id in item_ids])

    assert np.isfinite(fitted).all()
    assert scores["2"] > max(scores["0"], scores["1"], scores["3"])
    assert rank._bt_scc_solution_is_certified(
        fitted,
        observed_wins,
        fit_wins,
        1e-6,
    )


def test_tiny_pseudo_condensation_newton_resolves_sparse_scc_hierarchy(
    tmp_path,
):
    rng = np.random.default_rng(2277)
    for _ in range(58):
        n_items = int(rng.integers(2, 25))
        item_ids = [str(index) for index in range(n_items)]
        edges = {
            (index, int(rng.integers(index)))
            for index in range(1, n_items)
        }
        density = float(rng.uniform(0.05, 0.35))
        for left in range(n_items):
            for right in range(left + 1, n_items):
                if rng.random() < density:
                    edges.add((left, right))

        outcomes = []
        for left, right in sorted(edges):
            state = int(rng.integers(3))
            if state == 0:
                outcomes.append((str(left), str(right), 1.0))
            elif state == 1:
                outcomes.append((str(right), str(left), 1.0))
            else:
                outcomes.extend(
                    (
                        (str(left), str(right), 3.0),
                        (str(right), str(left), 1.0),
                    )
                )
        pseudo = float(10.0 ** rng.uniform(-300.0, 3.0))

    assert n_items == 16
    assert pseudo == pytest.approx(1.6510907983669315e-227)
    rank = _rank(tmp_path)
    scores = rank._fit_bt(
        item_ids,
        outcomes,
        pseudo=pseudo,
        max_iter=1000,
        tol=1e-6,
    )
    observed_wins, fit_wins = rank._build_bt_win_matrices(
        item_ids, outcomes, pseudo
    )
    fitted = np.array([scores[item_id] for item_id in item_ids])

    assert np.isfinite(fitted).all()
    assert rank._bt_scc_solution_is_certified(
        fitted,
        observed_wins,
        fit_wins,
        1e-6,
    )


def test_observed_information_has_graph_laplacian_structure(tmp_path):
    rank = _rank(tmp_path)
    item_ids = ["a", "b", "c"]
    outcomes = [
        ("a", "b", 3.0),
        ("b", "a", 2.0),
        ("b", "c", 4.0),
        ("c", "b", 1.0),
        ("c", "a", 2.0),
        ("a", "c", 2.0),
    ]
    scores, n_ij, p_ij = _fit(rank, item_ids, outcomes)
    q_ij = n_ij * p_ij * (1.0 - p_ij)
    fisher = np.diag(q_ij.sum(axis=1)) - q_ij

    np.testing.assert_allclose(fisher, fisher.T, atol=1e-12)
    np.testing.assert_allclose(fisher @ np.ones(len(item_ids)), 0.0, atol=1e-12)
    assert np.all(fisher[np.triu_indices(len(item_ids), 1)] < 0)
    eigenvalues = np.linalg.eigvalsh(fisher)
    assert np.sum(eigenvalues < 1e-10) == 1

    regularized_q = (n_ij + 0.1 * (n_ij > 0)) * p_ij * (1.0 - p_ij)
    bread = np.diag(regularized_q.sum(axis=1)) - regularized_q
    bread_inverse = np.linalg.pinv(bread, rcond=1e-12, hermitian=True)
    expected_covariance = bread_inverse @ fisher @ bread_inverse
    se = _standard_errors(rank, item_ids, scores, n_ij, p_ij)
    np.testing.assert_allclose(
        se, np.sqrt(np.diag(expected_covariance)), atol=1e-10
    )


def test_unregularized_fit_rejects_ford_condition_failure(tmp_path):
    rank = _rank(tmp_path)
    outcomes = [
        ("a", "b"),
        ("b", "a"),
        ("c", "d"),
        ("d", "c"),
        ("a", "c"),
    ]

    with pytest.raises(ValueError, match="directed win graph"):
        rank._fit_bt(
            ["a", "b", "c", "d"],
            outcomes,
            pseudo=0.0,
            max_iter=20_000,
            tol=1e-12,
        )


def test_abstentions_add_no_evidence_or_precision(tmp_path):
    rank = _rank(tmp_path, insufficient_signal_policy="abstain")
    base_outcomes = [("a", "b", 6.0), ("b", "a", 4.0)]
    abstentions = []
    for _ in range(100):
        _, decoded = rank._decode_pairwise_outcome(
            "a", "b", "insufficient signal"
        )
        abstentions.extend(decoded)

    base_scores, base_n, base_p = _fit(rank, ["a", "b"], base_outcomes)
    augmented_scores, augmented_n, augmented_p = _fit(
        rank, ["a", "b"], base_outcomes + abstentions
    )
    base_se = _standard_errors(rank, ["a", "b"], base_scores, base_n, base_p)
    augmented_se = _standard_errors(
        rank, ["a", "b"], augmented_scores, augmented_n, augmented_p
    )

    assert augmented_scores == pytest.approx(base_scores, abs=1e-12)
    np.testing.assert_allclose(augmented_n, base_n)
    np.testing.assert_allclose(augmented_se, base_se, atol=1e-12)


def test_isolated_items_do_not_change_scores_or_precision(tmp_path):
    rank = _rank(tmp_path)
    outcomes = [("a", "b", 6.0), ("b", "a", 4.0)]

    base_ids = ["a", "b"]
    base_scores, base_n, base_p = _fit(rank, base_ids, outcomes)
    base_se = _standard_errors(rank, base_ids, base_scores, base_n, base_p)

    extended_ids = base_ids + [f"unused-{i}" for i in range(25)]
    extended_scores, extended_n, extended_p = _fit(
        rank, extended_ids, outcomes
    )
    with pytest.warns(RuntimeWarning, match="comparison graph is disconnected"):
        extended_se = _standard_errors(
            rank, extended_ids, extended_scores, extended_n, extended_p
        )

    assert extended_scores["a"] - extended_scores["b"] == pytest.approx(
        base_scores["a"] - base_scores["b"], abs=1e-12
    )
    np.testing.assert_allclose(extended_n[:2, :2], base_n)
    np.testing.assert_allclose(extended_se[:2], base_se, atol=1e-12)
    assert np.isnan(extended_se[2:]).all()


def test_zscores_are_normalized_within_comparison_components(tmp_path):
    rank = _rank(tmp_path)
    base = rank._component_zscores(
        np.array([-1.0, 0.0, 1.0]), np.array([0, 0, 0])
    )
    extended = rank._component_zscores(
        np.array([-1.0, 0.0, 1.0, -100.0, 100.0]),
        np.array([0, 0, 0, 1, 1]),
    )

    np.testing.assert_allclose(extended[:3], base, atol=1e-12)
    np.testing.assert_allclose(extended[3:], [-1.0, 1.0], atol=1e-12)
    singleton = rank._component_zscores(np.array([0.0]), np.array([0]))
    assert np.isnan(singleton[0])


def test_invalid_fit_inputs_fail_loudly(tmp_path):
    rank = _rank(tmp_path)

    with pytest.raises(ValueError, match="unique"):
        _fit(rank, ["a", "a"], [("a", "a")])
    with pytest.raises(ValueError, match="non-negative"):
        _fit(rank, ["a", "b"], [("a", "b", -1.0)])
    with pytest.raises(ValueError, match="present in item_ids"):
        _fit(rank, ["a", "b"], [("a", "missing")])
    with pytest.raises(ValueError, match="pseudo"):
        _fit(rank, ["a", "b"], [("a", "b")], pseudo=-0.1)
    with pytest.raises(ValueError, match="learning_rate"):
        _fit(rank, ["a", "b"], [("a", "b")], pseudo=0.0)


def test_slow_mm_fit_is_polished_before_inference(tmp_path):
    rank = _rank(tmp_path)
    outcomes = [("a", "b")] * 9 + [("b", "a")]

    scores = rank._fit_bt(
        ["a", "b"],
        outcomes,
        pseudo=0.1,
        max_iter=1,
        tol=1e-15,
    )
    expected_gap = np.log((9.0 + 0.05) / (1.0 + 0.05))
    assert scores["a"] - scores["b"] == pytest.approx(
        expected_gap, abs=1e-10
    )


def test_production_solver_converges_on_sparse_directed_chain(tmp_path):
    rank = _rank(tmp_path)
    item_ids = [str(index) for index in range(50)]
    outcomes = [
        (str(index), str(index + 1)) for index in range(len(item_ids) - 1)
    ]

    scores, n_ij, p_ij = rank._fit_bt(
        item_ids,
        outcomes,
        pseudo=0.1,
        max_iter=rank._MAX_ITER,
        tol=rank._TOL,
        return_info=True,
    )
    score_values = np.array([scores[item_id] for item_id in item_ids])
    adjacent_gaps = score_values[:-1] - score_values[1:]
    np.testing.assert_allclose(adjacent_gaps, np.log(21.0), atol=2e-6)
    standard_errors = rank._bt_standard_errors(
        score_values,
        n_ij,
        p_ij,
        rcond=rank._SE_EIGEN_TOL,
        regularization_strength=0.1,
    )
    assert np.isfinite(standard_errors).all()


def test_live_rank_path_uses_fractional_draws(monkeypatch, tmp_path):
    async def fake_get_all_responses(*, identifiers, **kwargs):
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    rank = Rank(
        RankConfig(
            attributes={"quality": ""},
            save_dir=str(tmp_path),
            file_name="rankings",
            initial_rating_pass=False,
            n_rounds=1,
            matches_per_round=1,
        )
    )

    result = asyncio.run(
        rank.run(
            pd.DataFrame({"text": ["first", "second"]}),
            column_name="text",
            reset_files=True,
        )
    )

    np.testing.assert_allclose(result["quality_raw"], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        result["quality_se"], [1.0 / 1.1, 1.0 / 1.1], atol=1e-10
    )
    assert result["quality_component"].tolist() == [0, 0]
    metadata = json.loads((tmp_path / "rankings_run_metadata.json").read_text())
    assert metadata["rank_estimator_version"] == 2
    assert metadata["insufficient_signal_policy"] == "tie"
    assert metadata["learning_rate"] == 0.1
    assert metadata["last_completed_round"] == 0

    diagnostics = pd.read_csv(tmp_path / "rankings_diagnostics.csv")
    assert diagnostics.to_dict(orient="records") == [
        {
            "attribute": "quality",
            "circle_count": 0,
            "square_count": 0,
            "draw_count": 1,
            "insufficient_signal_count": 0,
            "invalid_count": 0,
            "effective_comparison_weight": 1.0,
            "comparison_components": 1,
            "isolated_items": 0,
            "finite_standard_errors": 2,
        }
    ]


def test_semantically_incomplete_response_is_not_committed(
    monkeypatch, tmp_path
):
    responses = ['{"quality": "draw"}', '{"quality": "draw", "novelty": "draw"}']
    calls = 0

    async def fake_get_all_responses(*, identifiers, **kwargs):
        nonlocal calls
        response = responses[min(calls, len(responses) - 1)]
        calls += 1
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": [response] * len(identifiers),
                "Successful": [True] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    cfg = RankConfig(
        attributes={"quality": "", "novelty": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    data = pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]})

    with pytest.raises(ValueError, match="semantically invalid"):
        asyncio.run(
            Rank(cfg).run(
                data,
                column_name="text",
                id_column="id",
                reset_files=True,
            )
        )
    metadata = json.loads((tmp_path / "rankings_run_metadata.json").read_text())
    assert metadata["last_completed_round"] == -1
    assert not (tmp_path / "rankings_round0.csv").exists()
    staged = pd.read_csv(tmp_path / ".rankings_round0.csv")
    assert not staged["Successful"].astype(bool).any()

    result = asyncio.run(
        Rank(cfg).run(
            data,
            column_name="text",
            id_column="id",
            reset_files=False,
        )
    )
    assert calls == 2
    assert result[["quality_raw", "novelty_raw"]].notna().all().all()


def test_duplicate_and_singleton_inputs_are_handled_before_model_calls(
    monkeypatch, tmp_path
):
    calls = 0

    async def fake_get_all_responses(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("no model call expected")

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path / "duplicate"),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    rank = Rank(cfg)

    with pytest.raises(ValueError, match="unique identifier"):
        asyncio.run(
            rank.run(
                pd.DataFrame({"id": [1, 1, 2], "text": ["a", "b", "c"]}),
                column_name="text",
                id_column="id",
                reset_files=True,
            )
        )

    singleton_rank = Rank(
        RankConfig(
            attributes={"quality": ""},
            save_dir=str(tmp_path / "singleton"),
            initial_rating_pass=False,
            n_rounds=1,
            matches_per_round=1,
        )
    )
    singleton = asyncio.run(
        singleton_rank.run(
            pd.DataFrame({"id": [7], "text": ["only"]}),
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )

    assert calls == 0
    assert singleton["quality_component"].tolist() == [0]
    assert singleton[["quality", "quality_raw", "quality_se"]].isna().all().all()


def test_abstention_resume_tracks_numeric_ids_and_preserves_round_rows(
    monkeypatch, tmp_path
):
    prompt_counts = []

    async def fake_get_all_responses(*, identifiers, **kwargs):
        prompt_counts.append(len(identifiers))
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": [
                    '{"quality": "insufficient signal"}'
                ]
                * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    first_cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        file_name="rankings",
        initial_rating_pass=False,
        insufficient_signal_policy="abstain",
        n_rounds=1,
        matches_per_round=1,
    )
    first_data = pd.DataFrame({"id": [101, 202], "text": ["a", "b"]})
    first = asyncio.run(
        Rank(first_cfg).run(
            first_data,
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )
    original_round = pd.read_csv(tmp_path / "rankings_round0.csv")
    original_rows = original_round.set_index("Identifier")[["Batch", "IdA", "IdB"]]
    call_boundary = len(prompt_counts)

    second_cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        file_name="rankings",
        initial_rating_pass=False,
        insufficient_signal_policy="abstain",
        n_rounds=2,
        matches_per_round=1,
    )
    expanded = asyncio.run(
        Rank(second_cfg).run(
            pd.DataFrame(
                {"id": [101, 202, 303], "text": ["a", "b", "c"]}
            ),
            column_name="text",
            id_column="id",
            reset_files=False,
        )
    )

    # Only the genuinely new numeric ID catches up; abstaining old pairs are
    # still recognized as completed comparisons even though they add no wins.
    assert prompt_counts[call_boundary:] == [1, 2]
    updated_round = pd.read_csv(tmp_path / "rankings_round0.csv").set_index(
        "Identifier"
    )
    pd.testing.assert_frame_equal(
        updated_round.loc[original_rows.index, ["Batch", "IdA", "IdB"]],
        original_rows,
        check_dtype=False,
    )
    assert first[["quality", "quality_raw", "quality_se"]].isna().all().all()
    assert expanded[["quality", "quality_raw", "quality_se"]].isna().all().all()
    assert expanded["quality_component"].nunique() == 3


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"rank_estimator_version": 1, "insufficient_signal_policy": "tie", "learning_rate": 0.1},
        {"rank_estimator_version": 3, "insufficient_signal_policy": "tie", "learning_rate": 0.1},
        {"rank_estimator_version": 2, "insufficient_signal_policy": "abstain", "learning_rate": 0.1},
        {"rank_estimator_version": 2, "insufficient_signal_policy": "tie", "learning_rate": True},
        {"rank_estimator_version": 2, "insufficient_signal_policy": "tie", "learning_rate": 1e-16},
    ],
)
def test_resume_rejects_incompatible_final_artifacts_before_calls(
    monkeypatch, tmp_path, metadata
):
    calls = 0

    async def fake_get_all_responses(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("incompatible cache must fail before model calls")

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    (tmp_path / "rankings_final.csv").write_text("text,quality\na,0\nb,0\n")
    (tmp_path / "rankings_run_metadata.json").write_text(json.dumps(metadata))
    rank = Rank(
        RankConfig(
            attributes={"quality": ""},
            save_dir=str(tmp_path),
            initial_rating_pass=False,
            n_rounds=1,
            matches_per_round=1,
        )
    )

    with pytest.raises(ValueError, match="incompatible"):
        asyncio.run(
            rank.run(
                pd.DataFrame({"text": ["a", "b"]}),
                column_name="text",
                reset_files=False,
            )
        )
    assert calls == 0


def test_recursive_rank_rejects_disconnected_pruning_scores(
    monkeypatch, tmp_path
):
    async def fake_get_all_responses(*, identifiers, **kwargs):
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": [
                    '{"quality": "insufficient signal"}'
                ]
                * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    rank = Rank(
        RankConfig(
            attributes={"quality": ""},
            save_dir=str(tmp_path),
            initial_rating_pass=False,
            insufficient_signal_policy="abstain",
            recursive=True,
            recursive_rate_first_round=False,
            recursive_fraction=0.75,
            recursive_min_remaining=1,
            n_rounds=1,
            matches_per_round=1,
        )
    )

    with pytest.raises(ValueError, match="disconnected comparison graph"):
        asyncio.run(
            rank.run(
                pd.DataFrame({"text": ["a", "b", "c", "d"]}),
                column_name="text",
                reset_files=True,
            )
        )


def test_resume_rejects_changed_payload_for_a_persisted_id(
    monkeypatch, tmp_path
):
    calls = 0

    async def fake_get_all_responses(*, identifiers, **kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    asyncio.run(
        Rank(cfg).run(
            pd.DataFrame({"id": ["a", "b"], "text": ["old a", "old b"]}),
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )
    call_boundary = calls

    with pytest.raises(ValueError, match="payload changed"):
        asyncio.run(
            Rank(cfg).run(
                pd.DataFrame(
                    {"id": ["a", "b"], "text": ["edited a", "old b"]}
                ),
                column_name="text",
                id_column="id",
                reset_files=False,
            )
        )
    assert calls == call_boundary


def test_round_replay_preserves_string_identifiers_that_resemble_csv_values(
    monkeypatch, tmp_path
):
    prompt_counts = []

    async def fake_get_all_responses(*, identifiers, **kwargs):
        prompt_counts.append(len(identifiers))
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    data = pd.DataFrame(
        {
            "id": ["001", "002", "NA", "null"],
            "text": ["a", "b", "c", "d"],
        }
    )
    first_cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    first = asyncio.run(
        Rank(first_cfg).run(
            data,
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )
    boundary = len(prompt_counts)
    second_cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=2,
        matches_per_round=1,
    )
    second = asyncio.run(
        Rank(second_cfg).run(
            data,
            column_name="text",
            id_column="id",
            reset_files=False,
        )
    )

    assert first["id"].tolist() == data["id"].tolist()
    assert second["id"].tolist() == data["id"].tolist()
    assert prompt_counts[boundary:] == [2]
    round_zero = Rank._read_rank_checkpoint(
        str(tmp_path / "rankings_round0.csv"), 1
    )
    assert set(round_zero["IdA"]) | set(round_zero["IdB"]) == set(data["id"])


def test_reset_clears_stale_future_rounds_and_batch_state(
    monkeypatch, tmp_path
):
    async def fake_get_all_responses(*, identifiers, **kwargs):
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    (tmp_path / "rankings_round4.csv").write_text("stale")
    (tmp_path / "rankings_round4.csv.batch_state.json").write_text("{}")
    result = asyncio.run(
        Rank(
            RankConfig(
                attributes={"quality": ""},
                save_dir=str(tmp_path),
                initial_rating_pass=False,
                n_rounds=1,
                matches_per_round=1,
            )
        ).run(
            pd.DataFrame({"text": ["a", "b"]}),
            column_name="text",
            reset_files=True,
        )
    )

    assert len(result) == 2
    assert not (tmp_path / "rankings_round4.csv").exists()
    assert not (tmp_path / "rankings_round4.csv.batch_state.json").exists()


def test_failed_model_row_does_not_commit_and_is_retried(monkeypatch, tmp_path):
    calls = 0

    async def fake_get_all_responses(*, identifiers, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return pd.DataFrame(
                {
                    "Identifier": identifiers,
                    "Response": [""] * len(identifiers),
                    "Successful": [False] * len(identifiers),
                }
            )
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "circle"}'] * len(identifiers),
                "Successful": [True] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    data = pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]})

    with pytest.raises(ValueError, match="not committed"):
        asyncio.run(
            Rank(cfg).run(
                data,
                column_name="text",
                id_column="id",
                reset_files=True,
            )
        )
    metadata = json.loads((tmp_path / "rankings_run_metadata.json").read_text())
    assert metadata["last_completed_round"] == -1

    result = asyncio.run(
        Rank(cfg).run(
            data,
            column_name="text",
            id_column="id",
            reset_files=False,
        )
    )

    assert calls == 2
    assert result["quality_raw"].notna().all()
    metadata = json.loads((tmp_path / "rankings_run_metadata.json").read_text())
    assert metadata["last_completed_round"] == 0


def test_committed_checkpoint_rejects_explicit_failure_or_blank_response(tmp_path):
    checkpoint = pd.DataFrame(
        {
            "Identifier": ["response-1"],
            "Response": [""],
            "Successful": [False],
            "Batch": [0],
            "Pair": [0],
            "IdA": ["a"],
            "IdB": ["b"],
        }
    )
    path = tmp_path / "rankings_round0.csv"
    checkpoint.to_csv(path, index=False)

    with pytest.raises(ValueError, match="failed or blank"):
        Rank._read_rank_checkpoint(str(path), 1)


def test_round_marker_write_failure_promotes_without_repeating_paid_work(
    monkeypatch, tmp_path
):
    calls = 0

    async def fake_get_all_responses(*, identifiers, **kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    rank_module = importlib.import_module("gabriel.tasks.rank")
    real_update = rank_module.update_run_metadata

    def fail_marker(*args, **kwargs):
        if kwargs.get("last_completed_round") == 0:
            raise OSError("simulated durable-marker failure")
        return real_update(*args, **kwargs)

    monkeypatch.setattr(rank_module, "get_all_responses", fake_get_all_responses)
    monkeypatch.setattr(rank_module, "update_run_metadata", fail_marker)
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    data = pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]})

    with pytest.raises(OSError, match="durable-marker"):
        asyncio.run(
            Rank(cfg).run(
                data,
                column_name="text",
                id_column="id",
                reset_files=True,
            )
        )
    metadata = json.loads((tmp_path / "rankings_run_metadata.json").read_text())
    assert metadata["last_completed_round"] == -1
    assert (tmp_path / "rankings_round0.csv").exists()

    monkeypatch.setattr(rank_module, "update_run_metadata", real_update)
    result = asyncio.run(
        Rank(cfg).run(
            data,
            column_name="text",
            id_column="id",
            reset_files=False,
        )
    )

    assert calls == 1
    assert result["quality_raw"].notna().all()
    metadata = json.loads((tmp_path / "rankings_run_metadata.json").read_text())
    assert metadata["last_completed_round"] == 0
    assert not (tmp_path / ".rankings_round0_plan.json").exists()
    assert not (tmp_path / ".rankings_round0.csv").exists()


def test_interrupted_round_plan_survives_input_row_reordering(
    monkeypatch, tmp_path
):
    collector_calls = 0
    paid_judgments = 0
    prompt_snapshots = []

    async def interrupted_collector(*, identifiers, prompts, save_path, **kwargs):
        nonlocal collector_calls, paid_judgments
        collector_calls += 1
        prompt_snapshots.append(list(prompts))
        if not Path(save_path).exists():
            paid_judgments += len(identifiers)
            pd.DataFrame(
                {
                    "Identifier": identifiers,
                    "Response": ['{"quality": "draw"}'] * len(identifiers),
                    "Successful": [True] * len(identifiers),
                }
            ).to_csv(save_path, index=False)
            raise RuntimeError("simulated crash after durable paid responses")
        return pd.read_csv(save_path)

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", interrupted_collector
    )
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    original = pd.DataFrame(
        {"id": ["a", "b", "c"], "text": ["A", "B", "C"]}
    )
    with pytest.raises(RuntimeError, match="durable paid"):
        asyncio.run(
            Rank(cfg).run(
                original,
                column_name="text",
                id_column="id",
                reset_files=True,
            )
        )

    metadata_path = tmp_path / "rankings_run_metadata.json"
    metadata_before_rejection = metadata_path.read_bytes()
    calls_before_rejection = collector_calls
    with pytest.raises(ValueError, match="different input set"):
        asyncio.run(
            Rank(cfg).run(
                pd.concat(
                    [
                        original,
                        pd.DataFrame({"id": ["g"], "text": ["old payload"]}),
                    ],
                    ignore_index=True,
                ),
                column_name="text",
                id_column="id",
                reset_files=False,
            )
        )
    assert collector_calls == calls_before_rejection
    assert metadata_path.read_bytes() == metadata_before_rejection

    result = asyncio.run(
        Rank(cfg).run(
            original.iloc[::-1].reset_index(drop=True),
            column_name="text",
            id_column="id",
            reset_files=False,
        )
    )

    assert collector_calls == 2
    assert paid_judgments == len(prompt_snapshots[0])
    assert prompt_snapshots[1] == prompt_snapshots[0]
    assert set(result["id"]) == {"a", "b", "c"}
    assert not (tmp_path / ".rankings_round0_plan.json").exists()
    assert not (tmp_path / ".rankings_round0.csv").exists()


def test_catchup_deduplicates_pair_deficits_and_uses_staging_file(
    monkeypatch, tmp_path
):
    calls = []

    async def fake_get_all_responses(*, identifiers, save_path, **kwargs):
        calls.append((len(identifiers), save_path))
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    first = Rank(cfg)
    first._generate_pairs = lambda *, texts_by_id, **kwargs: [
        (("a", texts_by_id["a"]), ("b", texts_by_id["b"])),
        (("c", texts_by_id["c"]), ("d", texts_by_id["d"])),
    ]
    asyncio.run(
        first.run(
            pd.DataFrame(
                {"id": ["a", "b", "c", "d"], "text": ["A", "B", "C", "D"]}
            ),
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )
    committed_before = (tmp_path / "rankings_round0.csv").read_bytes()

    result = asyncio.run(
        Rank(cfg).run(
            pd.DataFrame({"id": ["a", "c"], "text": ["A", "C"]}),
            column_name="text",
            id_column="id",
            reset_files=False,
        )
    )

    assert calls[0][0] == 2
    assert calls[1][0] == 1
    assert calls[0][1].endswith("rankings_round0.csv")
    assert calls[1][1].endswith(".rankings_catchup_round0.csv")
    assert (tmp_path / "rankings_round0.csv").read_bytes() != committed_before
    checkpoint = Rank._read_rank_checkpoint(
        str(tmp_path / "rankings_round0.csv"), 1
    )
    assert len(checkpoint) == 3
    assert result["quality_raw"].notna().all()
    assert result["quality_component"].nunique() == 1


def test_interrupted_catchup_cannot_mutate_committed_round(monkeypatch, tmp_path):
    phase = "initial"

    async def fake_get_all_responses(*, identifiers, save_path, **kwargs):
        if phase == "catchup":
            pd.DataFrame(
                {
                    "Identifier": identifiers[:1],
                    "Response": ['{"quality": "draw"}'],
                    "Successful": [True],
                }
            ).to_csv(save_path, index=False)
            raise RuntimeError("simulated interrupted catch-up")
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    first = Rank(cfg)
    first._generate_pairs = lambda *, texts_by_id, **kwargs: [
        (("a", texts_by_id["a"]), ("b", texts_by_id["b"])),
        (("c", texts_by_id["c"]), ("d", texts_by_id["d"])),
    ]
    asyncio.run(
        first.run(
            pd.DataFrame(
                {"id": ["a", "b", "c", "d"], "text": ["A", "B", "C", "D"]}
            ),
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )
    committed_path = tmp_path / "rankings_round0.csv"
    committed_before = committed_path.read_bytes()
    phase = "catchup"

    with pytest.raises(RuntimeError, match="interrupted catch-up"):
        asyncio.run(
            Rank(cfg).run(
                pd.DataFrame({"id": ["a", "c"], "text": ["A", "C"]}),
                column_name="text",
                id_column="id",
                reset_files=False,
            )
        )

    assert committed_path.read_bytes() == committed_before
    Rank._read_rank_checkpoint(str(committed_path), 1)
    assert (tmp_path / ".rankings_catchup_round0.csv").exists()


def test_empty_view_preserves_existing_tournament_and_handles_empty_ids(
    monkeypatch, tmp_path
):
    calls = 0

    async def fake_get_all_responses(*, identifiers, **kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    data = pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]})
    asyncio.run(
        Rank(cfg).run(
            data,
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )
    paths = [
        tmp_path / "rankings_run_metadata.json",
        tmp_path / "rankings_round0.csv",
        tmp_path / "rankings_final.csv",
        tmp_path / "rankings_diagnostics.csv",
    ]
    snapshots = {path: path.read_bytes() for path in paths}

    empty = asyncio.run(
        Rank(cfg).run(
            pd.DataFrame({"id": pd.Series(dtype=str), "text": pd.Series(dtype=str)}),
            column_name="text",
            id_column="id",
            reset_files=False,
        )
    )

    assert empty.empty
    assert calls == 1
    assert {path: path.read_bytes() for path in paths} == snapshots
    replayed = asyncio.run(
        Rank(cfg).run(
            data,
            column_name="text",
            id_column="id",
            reset_files=False,
        )
    )
    assert calls == 1
    assert len(replayed) == 2


def test_blank_identifier_is_dropped_before_any_model_call(monkeypatch, tmp_path):
    calls = 0

    async def fake_get_all_responses(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("a singleton should not call the model")

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    result = asyncio.run(
        Rank(
            RankConfig(
                attributes={"quality": "", "novelty": ""},
                save_dir=str(tmp_path),
                initial_rating_pass=False,
                n_rounds=1,
                matches_per_round=1,
            )
        ).run(
            pd.DataFrame({"id": ["   ", "b"], "text": ["blank", "B"]}),
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )

    assert calls == 0
    assert result["id"].tolist() == ["b"]


def test_local_media_content_change_invalidates_persisted_judgments(
    monkeypatch, tmp_path
):
    calls = 0

    async def fake_get_all_responses(*, identifiers, **kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    image_a = tmp_path / "a.img"
    image_b = tmp_path / "b.img"
    image_a.write_bytes(b"first image bytes")
    image_b.write_bytes(b"second image bytes")
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path / "run"),
        modality="image",
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    data = pd.DataFrame(
        {"id": ["a", "b"], "image": [str(image_a), str(image_b)]}
    )
    asyncio.run(
        Rank(cfg).run(
            data,
            column_name="image",
            id_column="id",
            reset_files=True,
        )
    )
    image_a.write_bytes(b"replacement bytes at the same path")

    with pytest.raises(ValueError, match="payload changed"):
        asyncio.run(
            Rank(cfg).run(
                data,
                column_name="image",
                id_column="id",
                reset_files=False,
            )
        )
    assert calls == 1


def test_recursive_empty_output_uses_documented_directory(tmp_path):
    result = asyncio.run(
        Rank(
            RankConfig(
                attributes={"quality": "", "novelty": ""},
                save_dir=str(tmp_path),
                recursive=True,
                recursive_rate_first_round=False,
                initial_rating_pass=False,
            )
        ).run(
            pd.DataFrame({"text": pd.Series(dtype=str)}),
            column_name="text",
            reset_files=True,
        )
    )

    assert result.empty
    assert {"quality", "novelty", "overall_rank", "exit_stage"}.issubset(
        result.columns
    )
    assert not any(
        column.endswith(("_raw", "_se")) for column in result.columns
    )
    assert (tmp_path / "rankings_recursive" / "recursive_final.csv").exists()


def test_recursive_empty_view_preserves_existing_final_and_empty_folder_is_new(
    monkeypatch, tmp_path
):
    calls = 0

    async def fake_get_all_responses(*, identifiers, **kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    recursive_folder = tmp_path / "rankings_recursive"
    recursive_folder.mkdir()
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        recursive=True,
        recursive_rate_first_round=False,
        recursive_min_remaining=2,
        recursive_final_round_multiplier=1,
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    asyncio.run(
        Rank(cfg).run(
            pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]}),
            column_name="text",
            id_column="id",
            reset_files=False,
        )
    )
    final_path = recursive_folder / "recursive_final.csv"
    snapshot = final_path.read_bytes()
    boundary = calls

    empty = asyncio.run(
        Rank(cfg).run(
            pd.DataFrame(
                {"id": pd.Series(dtype=str), "text": pd.Series(dtype=str)}
            ),
            column_name="text",
            id_column="id",
            reset_files=False,
        )
    )

    assert empty.empty
    assert calls == boundary
    assert final_path.read_bytes() == snapshot


def test_measurement_fingerprint_guards_attributes_without_sidecars(
    monkeypatch, tmp_path
):
    prompt_batches = []

    async def fake_get_all_responses(*, identifiers, prompts, **kwargs):
        prompt_batches.append(list(prompts))
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    data = pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]})
    original_cfg = RankConfig(
        attributes={"quality": "OLD_UNIQUE_DEFINITION"},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    asyncio.run(
        Rank(original_cfg).run(
            data,
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )
    for sidecar in (tmp_path / "attributes.json", tmp_path / "rankings_attrs.json"):
        sidecar.unlink()
    boundary = len(prompt_batches)

    incompatible_cfg = RankConfig(
        attributes={"quality": "NEW_UNIQUE_DEFINITION"},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=2,
        matches_per_round=1,
    )
    with pytest.raises(ValueError, match="measurement_spec_fingerprint"):
        asyncio.run(
            Rank(incompatible_cfg).run(
                data,
                column_name="text",
                id_column="id",
                reset_files=False,
            )
        )
    assert len(prompt_batches) == boundary
    assert not (tmp_path / "attributes.json").exists()
    assert not (tmp_path / "rankings_attrs.json").exists()

    compatible_cfg = RankConfig(
        attributes={"quality": "OLD_UNIQUE_DEFINITION"},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=2,
        matches_per_round=1,
    )
    asyncio.run(
        Rank(compatible_cfg).run(
            data,
            column_name="text",
            id_column="id",
            reset_files=False,
        )
    )

    assert len(prompt_batches) == boundary + 1
    assert all("OLD_UNIQUE_DEFINITION" in prompt for prompt in prompt_batches[-1])
    assert (tmp_path / "attributes.json").exists()
    assert (tmp_path / "rankings_attrs.json").exists()


def test_recursive_orchestrator_guards_attributes_without_top_sidecars(
    monkeypatch, tmp_path
):
    calls = 0

    async def fake_get_all_responses(*, identifiers, **kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    data = pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]})
    original_cfg = RankConfig(
        attributes={"quality": "OLD_RECURSIVE_DEFINITION"},
        save_dir=str(tmp_path),
        recursive=True,
        recursive_rate_first_round=False,
        recursive_min_remaining=2,
        recursive_final_round_multiplier=1,
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    asyncio.run(
        Rank(original_cfg).run(
            data,
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )
    for sidecar in (tmp_path / "attributes.json", tmp_path / "rankings_attrs.json"):
        sidecar.unlink()
    boundary = calls

    incompatible_cfg = RankConfig(
        attributes={"replacement": "NEW_RECURSIVE_DEFINITION"},
        save_dir=str(tmp_path),
        recursive=True,
        recursive_rate_first_round=False,
        recursive_min_remaining=2,
        recursive_final_round_multiplier=1,
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    with pytest.raises(ValueError, match="measurement_spec_fingerprint"):
        asyncio.run(
            Rank(incompatible_cfg).run(
                data,
                column_name="text",
                id_column="id",
                reset_files=False,
            )
        )

    assert calls == boundary
    assert not (tmp_path / "attributes.json").exists()
    assert not (tmp_path / "rankings_attrs.json").exists()


def test_missing_persisted_input_fingerprints_fail_closed(monkeypatch, tmp_path):
    calls = 0

    async def fake_get_all_responses(*, identifiers, **kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    original = pd.DataFrame({"id": ["a", "b"], "text": ["original A", "B"]})
    asyncio.run(
        Rank(cfg).run(
            original,
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )
    metadata_path = tmp_path / "rankings_run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["input_fingerprints"] = {}
    metadata_path.write_text(json.dumps(metadata))
    boundary = calls

    with pytest.raises(ValueError, match="lacks valid content fingerprints"):
        asyncio.run(
            Rank(cfg).run(
                pd.DataFrame(
                    {"id": ["a", "b"], "text": ["changed A", "B"]}
                ),
                column_name="text",
                id_column="id",
                reset_files=False,
            )
        )
    assert calls == boundary


def test_recursive_top_fingerprint_detects_local_media_change(
    monkeypatch, tmp_path
):
    calls = 0

    async def fake_rate_responses(*, identifiers, **kwargs):
        nonlocal calls
        calls += len(identifiers)
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": 50}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rate.get_all_responses", fake_rate_responses
    )
    image_a = tmp_path / "a.img"
    image_b = tmp_path / "b.img"
    image_a.write_bytes(b"image a version one")
    image_b.write_bytes(b"image b")
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path / "run"),
        modality="image",
        recursive=True,
        recursive_rate_first_round=True,
        recursive_min_remaining=2,
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    data = pd.DataFrame(
        {"id": ["a", "b"], "image": [str(image_a), str(image_b)]}
    )
    asyncio.run(
        Rank(cfg).run(
            data,
            column_name="image",
            id_column="id",
            reset_files=True,
        )
    )
    boundary = calls
    image_a.write_bytes(b"image a replacement at same path")

    with pytest.raises(ValueError, match="comparison payload changed"):
        asyncio.run(
            Rank(cfg).run(
                data,
                column_name="image",
                id_column="id",
                reset_files=False,
            )
        )
    assert calls == boundary


def test_recursive_rate_semantics_are_in_top_fingerprint(monkeypatch, tmp_path):
    models = []

    async def fake_rate_responses(*, identifiers, model, **kwargs):
        models.extend([model] * len(identifiers))
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": 50}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rate.get_all_responses", fake_rate_responses
    )
    data = pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]})
    first_cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        recursive=True,
        recursive_rate_first_round=True,
        recursive_min_remaining=2,
        initial_rating_pass=False,
        rate_kwargs={"model": "rate-model-A"},
        n_rounds=1,
        matches_per_round=1,
    )
    asyncio.run(
        Rank(first_cfg).run(
            data,
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )
    boundary = len(models)

    changed_cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        recursive=True,
        recursive_rate_first_round=True,
        recursive_min_remaining=2,
        initial_rating_pass=False,
        rate_kwargs={"model": "rate-model-B"},
        n_rounds=1,
        matches_per_round=1,
    )
    with pytest.raises(ValueError, match="measurement_spec_fingerprint"):
        asyncio.run(
            Rank(changed_cfg).run(
                data,
                column_name="text",
                id_column="id",
                reset_files=False,
            )
        )

    assert len(models) == boundary
    assert set(models) == {"rate-model-A"}


def test_initial_rate_crash_still_persists_top_measurement_guard(
    monkeypatch, tmp_path
):
    models = []

    async def fake_rate_responses(*, identifiers, model, **kwargs):
        models.extend([model] * len(identifiers))
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": 50}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rate.get_all_responses", fake_rate_responses
    )
    first_cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=True,
        rate_kwargs={"model": "rate-model-A"},
        n_rounds=1,
        matches_per_round=1,
    )
    first = Rank(first_cfg)

    def crash_after_rate(*args, **kwargs):
        raise RuntimeError("simulated crash after initial Rate")

    monkeypatch.setattr(first, "_seed_ratings_from_rate", crash_after_rate)
    data = pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]})
    with pytest.raises(RuntimeError, match="after initial Rate"):
        asyncio.run(
            first.run(
                data,
                column_name="text",
                id_column="id",
                reset_files=True,
            )
        )
    metadata = json.loads((tmp_path / "rankings_run_metadata.json").read_text())
    assert metadata["last_completed_round"] == -1
    boundary = len(models)

    changed_cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=True,
        rate_kwargs={"model": "rate-model-B"},
        n_rounds=1,
        matches_per_round=1,
    )
    with pytest.raises(ValueError, match="measurement_spec_fingerprint"):
        asyncio.run(
            Rank(changed_cfg).run(
                data,
                column_name="text",
                id_column="id",
                reset_files=False,
            )
        )
    assert len(models) == boundary


def test_batch_state_only_round_is_guarded_before_resume_calls(
    monkeypatch, tmp_path
):
    models = []

    async def interrupted_batch(*, save_path, model, **kwargs):
        models.append(model)
        Path(f"{save_path}.batch_state.json").write_text(
            json.dumps({"batches": []})
        )
        raise RuntimeError("simulated crash after batch submission")

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", interrupted_batch
    )
    data = pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]})
    first_cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        model="model-A",
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    with pytest.raises(RuntimeError, match="after batch submission"):
        asyncio.run(
            Rank(first_cfg).run(
                data,
                column_name="text",
                id_column="id",
                reset_files=True,
            )
        )

    changed_cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        model="model-B",
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    with pytest.raises(ValueError, match="measurement_spec_fingerprint"):
        asyncio.run(
            Rank(changed_cfg).run(
                data,
                column_name="text",
                id_column="id",
                reset_files=False,
            )
        )

    with pytest.raises(ValueError, match="without a durable response"):
        asyncio.run(
            Rank(first_cfg).run(
                data,
                column_name="text",
                id_column="id",
                reset_files=False,
            )
        )
    assert models == ["model-A"]


def test_submitted_batch_with_round_plan_reenters_recovery_collector(
    monkeypatch, tmp_path
):
    calls = 0

    async def interrupted_batch(*, identifiers, save_path, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            pd.DataFrame(columns=["Identifier", "Response"]).to_csv(
                save_path, index=False
            )
            Path(f"{save_path}.batch_state.json").write_text(
                json.dumps(
                    {
                        "batches": [
                            {
                                "batch_id": "batch-paid-1",
                                "status": "submitted",
                                "total": len(identifiers),
                            }
                        ]
                    }
                )
            )
            raise RuntimeError("simulated crash after batch submission")
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
                "Successful": [True] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", interrupted_batch
    )
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    data = pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]})
    with pytest.raises(RuntimeError, match="after batch submission"):
        asyncio.run(
            Rank(cfg).run(
                data,
                column_name="text",
                id_column="id",
                reset_files=True,
            )
        )

    result = asyncio.run(
        Rank(cfg).run(
            data,
            column_name="text",
            id_column="id",
            reset_files=False,
        )
    )
    assert calls == 2
    assert result["quality_raw"].notna().all()
    assert not (tmp_path / ".rankings_round0.csv.batch_state.json").exists()


def test_resume_rejects_fewer_rounds_than_already_committed(
    monkeypatch, tmp_path
):
    calls = 0

    async def fake_get_all_responses(*, identifiers, **kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    data = pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]})
    asyncio.run(
        Rank(
            RankConfig(
                attributes={"quality": ""},
                save_dir=str(tmp_path),
                initial_rating_pass=False,
                n_rounds=2,
                matches_per_round=1,
            )
        ).run(
            data,
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )
    boundary = calls

    with pytest.raises(ValueError, match="fewer n_rounds"):
        asyncio.run(
            Rank(
                RankConfig(
                    attributes={"quality": ""},
                    save_dir=str(tmp_path),
                    initial_rating_pass=False,
                    n_rounds=1,
                    matches_per_round=1,
                )
            ).run(
                data,
                column_name="text",
                id_column="id",
                reset_files=False,
            )
        )
    assert calls == boundary


def test_resume_rejects_changed_runtime_judgment_settings_before_new_calls(
    monkeypatch, tmp_path
):
    calls = 0

    async def fake_get_all_responses(*, identifiers, **kwargs):
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    data = pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]})
    first_cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    asyncio.run(
        Rank(first_cfg).run(
            data,
            column_name="text",
            id_column="id",
            reset_files=True,
            web_search=False,
        )
    )
    boundary = calls

    resumed_cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=2,
        matches_per_round=1,
    )
    with pytest.raises(ValueError, match="measurement_spec_fingerprint"):
        asyncio.run(
            Rank(resumed_cfg).run(
                data,
                column_name="text",
                id_column="id",
                reset_files=False,
                web_search=True,
            )
        )
    assert calls == boundary

    task = Rank(first_cfg)
    assert task._measurement_spec_fingerprint(
        {
            "image_detail": "low",
            "api_key": "first-secret",
            "skip_tail_fails": True,
            "timeout": 1,
            "return_raw": False,
            "request_phase_callback": lambda *_: None,
        }
    ) == task._measurement_spec_fingerprint(
        {
            "image_detail": "low",
            "api_key": "second-secret",
            "skip_tail_fails": False,
            "timeout": 120,
            "return_raw": True,
            "request_phase_callback": lambda *_: None,
        }
    )
    assert task._measurement_spec_fingerprint(
        {"image_detail": "low"}
    ) != task._measurement_spec_fingerprint({"image_detail": "high"})


def test_custom_rank_judge_requires_explicit_version_before_calls(tmp_path):
    calls = 0

    def make_judge(label):
        async def judge(*, identifiers, **kwargs):
            nonlocal calls
            calls += 1
            return pd.DataFrame(
                {
                    "Identifier": identifiers,
                    "Response": [json.dumps({"quality": label})]
                    * len(identifiers),
                }
            )

        return judge

    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    with pytest.raises(ValueError, match="judge_version is required"):
        asyncio.run(
            Rank(cfg).run(
                pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]}),
                column_name="text",
                id_column="id",
                reset_files=True,
                get_all_responses_fn=make_judge("draw"),
            )
        )
    assert calls == 0


def test_recursive_rejected_superset_does_not_mutate_top_metadata(
    monkeypatch, tmp_path
):
    calls = 0

    async def interrupted_collector(*, identifiers, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("simulated recursive stage interruption")

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", interrupted_collector
    )
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        recursive=True,
        recursive_rate_first_round=False,
        recursive_fraction=0.5,
        recursive_min_remaining=1,
        n_rounds=1,
        matches_per_round=1,
    )
    original = pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]})
    with pytest.raises(RuntimeError, match="recursive stage interruption"):
        asyncio.run(
            Rank(cfg).run(
                original,
                column_name="text",
                id_column="id",
                reset_files=True,
            )
        )
    metadata_path = tmp_path / "rankings_run_metadata.json"
    metadata_before = metadata_path.read_bytes()
    call_boundary = calls

    with pytest.raises(ValueError, match="unfinished stage transaction"):
        asyncio.run(
            Rank(cfg).run(
                pd.concat(
                    [
                        original,
                        pd.DataFrame({"id": ["g"], "text": ["old payload"]}),
                    ],
                    ignore_index=True,
                ),
                column_name="text",
                id_column="id",
                reset_files=False,
            )
        )
    assert calls == call_boundary
    assert metadata_path.read_bytes() == metadata_before


def test_rank_forces_exact_tail_retries_for_rounds_and_catchup(
    monkeypatch, tmp_path
):
    tail_settings = []

    async def fake_get_all_responses(*, identifiers, **kwargs):
        tail_settings.append(kwargs.get("skip_tail_fails"))
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    asyncio.run(
        Rank(cfg).run(
            pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]}),
            column_name="text",
            id_column="id",
            reset_files=True,
            skip_tail_fails=True,
        )
    )
    asyncio.run(
        Rank(cfg).run(
            pd.DataFrame(
                {"id": ["a", "b", "c"], "text": ["A", "B", "C"]}
            ),
            column_name="text",
            id_column="id",
            reset_files=False,
            skip_tail_fails=True,
        )
    )
    assert len(tail_settings) >= 2
    assert tail_settings == [False] * len(tail_settings)


@pytest.mark.parametrize("recursive", [False, True])
def test_reset_refuses_to_orphan_durable_batch_state(tmp_path, recursive):
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        recursive=recursive,
        recursive_rate_first_round=False,
        n_rounds=1,
        matches_per_round=1,
    )
    if recursive:
        state_path = (
            tmp_path
            / "rankings_recursive"
            / "stage1"
            / ".rankings_round0.csv.batch_state.json"
        )
    else:
        state_path = tmp_path / ".rankings_round0.csv.batch_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "batches": [
                    {"batch_id": "batch-paid", "status": "in_progress"}
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="cannot reset.*Batch API state"):
        asyncio.run(
            Rank(cfg).run(
                pd.DataFrame({"id": [], "text": []}),
                column_name="text",
                id_column="id",
                reset_files=True,
            )
        )
    assert state_path.exists()


def test_reset_refuses_active_initial_rate_batch_state(tmp_path):
    state_path = (
        tmp_path
        / "rankings_initial_rate"
        / "rankings_initial_rate_raw_responses.csv.batch_state.json"
    )
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {"batches": [{"batch_id": "paid-rate", "status": "in_progress"}]}
        )
    )
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=True,
        n_rounds=1,
        matches_per_round=1,
    )

    with pytest.raises(ValueError, match="cannot reset.*Batch API state"):
        asyncio.run(
            Rank(cfg).run(
                pd.DataFrame({"id": [], "text": []}),
                column_name="text",
                id_column="id",
                reset_files=True,
            )
        )
    assert state_path.exists()


def test_recursive_reset_clears_task_owned_stage_tree(tmp_path):
    recursive_dir = tmp_path / "rankings_recursive"
    sentinel = recursive_dir / "stage9" / "stale.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("stale")
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        recursive=True,
        recursive_rate_first_round=False,
        n_rounds=1,
        matches_per_round=1,
    )
    task = Rank(cfg)

    async def fake_recursive(df, column_name, **kwargs):
        assert not sentinel.exists()
        return df.copy()

    task._run_recursive = fake_recursive
    result = asyncio.run(
        task.run(
            pd.DataFrame({"id": ["a"], "text": ["A"]}),
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )
    assert list(result["id"]) == ["a"]
    assert not sentinel.exists()


def test_catchup_avoids_repeating_committed_survivor_pairs(monkeypatch, tmp_path):
    prompt_counts = []

    async def fake_get_all_responses(*, identifiers, **kwargs):
        prompt_counts.append(len(identifiers))
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": ['{"quality": "draw"}'] * len(identifiers),
            }
        )

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=2,
    )
    first = Rank(cfg)
    original_pairs = [
        ("a", "b"),
        ("c", "d"),
        ("a", "x"),
        ("b", "z"),
        ("c", "y"),
        ("d", "w"),
    ]
    first._generate_pairs = lambda *, texts_by_id, **kwargs: [
        ((id_a, texts_by_id[id_a]), (id_b, texts_by_id[id_b]))
        for id_a, id_b in original_pairs
    ]
    full_ids = ["a", "b", "c", "d", "x", "z", "y", "w"]
    asyncio.run(
        first.run(
            pd.DataFrame({"id": full_ids, "text": [value.upper() for value in full_ids]}),
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )
    resumed = Rank(cfg)
    resumed.rng.choice = lambda values: values[0]
    survivors = ["a", "b", "c", "d"]
    result = asyncio.run(
        resumed.run(
            pd.DataFrame(
                {"id": survivors, "text": [value.upper() for value in survivors]}
            ),
            column_name="text",
            id_column="id",
            reset_files=False,
        )
    )

    assert prompt_counts == [6, 2]
    checkpoint = Rank._read_rank_checkpoint(
        str(tmp_path / "rankings_round0.csv"), 1
    )
    current_pairs = [
        tuple(sorted((id_a, id_b)))
        for id_a, id_b in zip(checkpoint["IdA"], checkpoint["IdB"])
        if id_a in survivors and id_b in survivors
    ]
    assert len(current_pairs) == len(set(current_pairs)) == 4
    assert {("a", "b"), ("c", "d")}.issubset(current_pairs)
    assert result["quality_component"].nunique() == 1


def test_interrupted_catchup_resumes_persisted_plan_without_orphaning_rows(
    monkeypatch, tmp_path
):
    phase = "initial"
    planned_ids = []
    paid_resume_calls = 0

    async def fake_get_all_responses(*, identifiers, save_path, **kwargs):
        nonlocal paid_resume_calls
        if phase == "initial":
            return pd.DataFrame(
                {
                    "Identifier": identifiers,
                    "Response": ['{"quality": "draw"}'] * len(identifiers),
                }
            )
        if phase == "interrupt":
            planned_ids[:] = list(identifiers)
            pd.DataFrame(
                {
                    "Identifier": identifiers[:1],
                    "Response": ['{"quality": "draw"}'],
                    "Successful": [True],
                }
            ).to_csv(save_path, index=False)
            raise RuntimeError("simulated partial catch-up")
        assert list(identifiers) == planned_ids
        staged = pd.read_csv(save_path)
        completed = set(staged["Identifier"].astype(str))
        missing = [identifier for identifier in identifiers if identifier not in completed]
        paid_resume_calls += len(missing)
        appended = pd.DataFrame(
            {
                "Identifier": missing,
                "Response": ['{"quality": "draw"}'] * len(missing),
                "Successful": [True] * len(missing),
            }
        )
        combined = pd.concat([staged, appended], ignore_index=True)
        combined.to_csv(save_path, index=False)
        return combined

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    initial = Rank(cfg)
    initial._generate_pairs = lambda *, texts_by_id, **kwargs: [
        (("a", texts_by_id["a"]), ("b", texts_by_id["b"]))
    ]
    asyncio.run(
        initial.run(
            pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]}),
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )
    expanded = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "text": ["A", "B", "C", "D", "E", "F"],
        }
    )
    phase = "interrupt"
    with pytest.raises(RuntimeError, match="partial catch-up"):
        asyncio.run(
            Rank(cfg).run(
                expanded,
                column_name="text",
                id_column="id",
                reset_files=False,
            )
        )
    assert len(planned_ids) == 2
    assert (tmp_path / ".rankings_catchup_round0_plan.json").exists()

    phase = "resume"
    metadata_path = tmp_path / "rankings_run_metadata.json"
    metadata_before_rejected_resume = metadata_path.read_bytes()
    with pytest.raises(ValueError, match="different input set"):
        asyncio.run(
            Rank(cfg).run(
                pd.concat(
                    [
                        expanded,
                        pd.DataFrame({"id": ["g"], "text": ["G"]}),
                    ],
                    ignore_index=True,
                ),
                column_name="text",
                id_column="id",
                reset_files=False,
            )
        )
    assert paid_resume_calls == 0
    assert metadata_path.read_bytes() == metadata_before_rejected_resume

    resumed = Rank(cfg)
    resumed.rng.choice = lambda values: values[-1]
    asyncio.run(
        resumed.run(
            expanded,
            column_name="text",
            id_column="id",
            reset_files=False,
        )
    )

    assert paid_resume_calls == 1
    checkpoint = Rank._read_rank_checkpoint(
        str(tmp_path / "rankings_round0.csv"), 1
    )
    assert set(planned_ids).issubset(set(checkpoint["Identifier"]))
    assert not (tmp_path / ".rankings_catchup_round0_plan.json").exists()
    assert not (tmp_path / ".rankings_catchup_round0.csv").exists()


def test_empty_catchup_batch_state_without_staging_fails_closed(
    monkeypatch, tmp_path
):
    phase = "initial"
    calls = 0

    async def fake_get_all_responses(*, identifiers, save_path, **kwargs):
        nonlocal calls
        calls += 1
        if phase == "initial":
            return pd.DataFrame(
                {
                    "Identifier": identifiers,
                    "Response": ['{"quality": "draw"}'] * len(identifiers),
                }
            )
        Path(f"{save_path}.batch_state.json").write_text(
            json.dumps({"batches": []})
        )
        raise RuntimeError("simulated post-batch pre-staging crash")

    monkeypatch.setattr(
        "gabriel.tasks.rank.get_all_responses", fake_get_all_responses
    )
    cfg = RankConfig(
        attributes={"quality": ""},
        save_dir=str(tmp_path),
        initial_rating_pass=False,
        n_rounds=1,
        matches_per_round=1,
    )
    asyncio.run(
        Rank(cfg).run(
            pd.DataFrame({"id": ["a", "b"], "text": ["A", "B"]}),
            column_name="text",
            id_column="id",
            reset_files=True,
        )
    )
    expanded = pd.DataFrame(
        {"id": ["a", "b", "c", "d"], "text": ["A", "B", "C", "D"]}
    )
    phase = "catchup"
    with pytest.raises(RuntimeError, match="pre-staging"):
        asyncio.run(
            Rank(cfg).run(
                expanded,
                column_name="text",
                id_column="id",
                reset_files=False,
            )
        )
    boundary = calls

    with pytest.raises(ValueError, match="without a durable response checkpoint"):
        asyncio.run(
            Rank(cfg).run(
                expanded,
                column_name="text",
                id_column="id",
                reset_files=False,
            )
        )
    assert calls == boundary
