import numpy as np
import pytest

from harness.metrics import MetricError, evaluate, evaluate_reference


@pytest.mark.parametrize("seed", range(20))
def test_numpy_metrics_match_organizer_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = 500
    users = rng.integers(0, 23, size=rows)
    labels = rng.integers(0, 2, size=rows)
    # Rounded scores deliberately exercise ties.
    scores = np.round(rng.normal(size=rows), 1)
    actual = evaluate(users, labels, scores)
    expected = evaluate_reference(users, labels, scores)
    for key in ("GAUC", "nDCG@5", "primary"):
        assert actual[key] == pytest.approx(expected[key], abs=1e-14)
    assert actual["users"] == expected["users"]
    assert actual["rows"] == expected["rows"]


def test_all_negative_users_count_for_ndcg_but_not_gauc() -> None:
    result = evaluate(
        np.array([1, 1, 2, 2]),
        np.array([0, 0, 0, 1]),
        np.array([0.8, 0.2, 0.1, 0.9]),
    )
    assert result["GAUC"] == 1.0
    assert result["nDCG@5"] == 0.5
    assert result["primary"] == 0.75


def test_rejects_nonfinite_or_nonbinary_input() -> None:
    with pytest.raises(MetricError, match="NaN"):
        evaluate([1], [1], [np.nan])
    with pytest.raises(MetricError, match="binary"):
        evaluate([1], [2], [0.0])
