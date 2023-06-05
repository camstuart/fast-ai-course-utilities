import numpy as np

from fast_ai_course_utilities.data_structures import scale_minmax


def test_scale_minmax() -> None:
    X = np.array([1, 2, 3, 4, 5])
    expected_result = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.array_equal(scale_minmax(X), expected_result)

    X = np.array([5, 10, 15, 20, 25])
    expected_result = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.array_equal(scale_minmax(X), expected_result)

    X = np.array([10, 20, 30, 40, 50])
    expected_result = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.array_equal(scale_minmax(X), expected_result)

