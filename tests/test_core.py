import numpy as np

from pcselect import calculate_optimal_pcs, get_variance_summary, visualize_pc_selection


def test_calculate_optimal_pcs_basic():
    stdev = np.array([4.0, 3.2, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 0.9, 0.8])
    rec = calculate_optimal_pcs(stdev)
    assert isinstance(rec, int)
    assert 1 <= rec <= len(stdev)


def test_get_variance_summary_shape():
    stdev = np.array([4.0, 3.2, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 0.9, 0.8])
    summary = get_variance_summary(stdev, pc_range=[3, 5, 10])
    assert list(summary.columns) == [
        "n_pcs",
        "cumulative_variance",
        "marginal_variance",
        "avg_variance_per_pc",
    ]
    assert summary.shape[0] == 3


def test_visualize_pc_selection_returns_figure_and_data():
    stdev = np.array([4.0, 3.2, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 0.9, 0.8])
    data, fig = visualize_pc_selection(stdev, max_pcs=10)
    assert data.shape[0] == 10
    assert fig is not None
