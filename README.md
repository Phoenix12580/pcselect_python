# pcselect (Python)

A Python implementation of the main logic from the R package [`scPCselect`](https://github.com/xiaoqqjun/scPCselect), used to select an appropriate number of principal components.

## Install

```bash
pip install -e .
```

## Usage

```python
import numpy as np
from pcselect import calculate_optimal_pcs, get_variance_summary, visualize_pc_selection

# Example stdev values from PCA
stdev = np.array([4.0, 3.2, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0, 0.9, 0.8])

recommended = calculate_optimal_pcs(stdev)
print(recommended)

summary = get_variance_summary(stdev, pc_range=[3, 5, 8, 10])
print(summary)

plot_data, fig = visualize_pc_selection(stdev, max_pcs=10)
fig.show()
```

## API

- `calculate_optimal_pcs(pca_like, min_variance=0.75, max_marginal_gain=0.005) -> int`
- `get_variance_summary(pca_like, pc_range=None) -> pandas.DataFrame`
- `visualize_pc_selection(pca_like, max_pcs=50, variance_thresholds=(0.6, 0.7, 0.8, 0.9)) -> (DataFrame, Figure)`

`pca_like` can be:
- a 1D array/iterable of PCA standard deviations;
- a scikit-learn PCA object (uses `explained_variance_`);
- an AnnData-like object with `uns['pca']['variance']`.
