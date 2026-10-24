"""Hand-calculated ranking examples and evaluation boundary cases."""

import pytest


from atelier_ai.ranking_metrics import evaluate_recommendations
from atelier_ai.ranking_metrics import score_recommendation_list


@pytest.mark.parametrize(
    "relevant,predictions,k,expected_ap,expected_recall",
    [
        (["a", "b", "c"], ["a", "x", "b"], 3, 5 / 9, 2 / 3),
        (["a", "b"], ["x", "y", "z"], 3, 0.0, 0.0),
        (["a", "b", "c", "d"], ["a", "b", "c"], 3, 1.0, 3 / 4),
        (["a", "a", "b"], ["a", "b", "x"], 3, 1.0, 1.0),
        (["a"], ["x", "y", "z", "a"], 3, 0.0, 0.0),
    ],
)
def test_scores_match_hand_calculated_examples(
    relevant, predictions, k, expected_ap, expected_recall
):
    result = score_recommendation_list(relevant, predictions, k=k)

    assert result["n_target_articles"] == len(set(relevant))
    assert result["ap_at_k"] == pytest.approx(expected_ap)
    assert result["recall_at_k"] == pytest.approx(expected_recall)


def test_ap_and_recall_use_different_denominators_above_twelve_targets():
    relevant = [f"target_{i}" for i in range(13)]
    predictions = ["target_0"] + [f"other_{i}" for i in range(11)]

    result = score_recommendation_list(relevant, predictions, k=12)

    assert result["n_hits"] == 1
    assert result["n_target_articles"] == 13
    assert result["ap_at_k"] == pytest.approx(1 / 12)
    assert result["recall_at_k"] == pytest.approx(1 / 13)


@pytest.mark.parametrize(
    "relevant,predictions,k",
    [
        ([], ["a", "b", "c"], 3),
        (["a"], ["a", "a", "b"], 3),
        (["a"], ["a", "b"], 3),
        (["a"], ["a"], 0),
    ],
)
def test_invalid_metric_inputs_are_rejected(relevant, predictions, k):
    with pytest.raises(ValueError):
        score_recommendation_list(relevant, predictions, k=k)

def evaluation_example():
    return {
        "recommendation_lists": {
            "u": ["a", "x"],
            "v": ["q", "r"],
            "w": ["m", "n"],
        },
        "relevant_articles": {"u": {"a", "b"}, "v": {"missed"}},
        "customer_ids": ["u", "v", "w"],
        "k": 2,
    }


def test_evaluation_preserves_missed_buyers_and_full_denominators():
    result = evaluate_recommendations(**evaluation_example())
    scores = result["customer_scores"].set_index("customer_id")
    summary = result["summary"]

    assert set(scores.index) == {"u", "v"}
    assert scores.loc["v", "ap_at_k"] == 0
    assert scores.loc["v", "recall_at_k"] == 0
    assert summary["recommendation_customers"] == 3
    assert summary["evaluated_customers"] == 2
    assert summary["excluded_customers"] == 1
    assert summary["excluded_share_pct"] == pytest.approx(100 / 3)
    assert summary["target_pairs"] == 3
    assert summary["retrieved_target_pairs"] == 1
    assert summary["map_at_k"] == pytest.approx(0.25)
    assert summary["recall_at_k"] == pytest.approx(0.25)
    assert summary["micro_recall_at_k"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "problem",
    ["missing_recipient", "invalid_nonbuyer_list", "unknown_reference_customer"],
)
def test_evaluation_rejects_population_and_list_errors(problem):
    arguments = evaluation_example()
    if problem == "missing_recipient":
        del arguments["recommendation_lists"]["w"]
    elif problem == "invalid_nonbuyer_list":
        arguments["recommendation_lists"]["w"] = ["m", "m"]
    else:
        arguments["relevant_articles"]["outside"] = {"a"}

    with pytest.raises(ValueError):
        evaluate_recommendations(**arguments)