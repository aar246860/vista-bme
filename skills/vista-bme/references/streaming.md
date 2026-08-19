# Streaming updates on fixed support

VISTA-BME provides three streaming utilities with different purposes. Choose one based on the retained state and scale.

## Repeated observations at fixed nodes

`StreamingSiteAccumulator` stores one count, likelihood precision, and natural mean per latent node. Its memory does not grow with the number of repeated Gaussian observations.

```python
import numpy as np

from stamps.bme import StreamingSiteAccumulator, build_vecchia_graph

coordinates = np.array([
    [0.0, 0.0],
    [0.5, 0.0],
    [1.0, 0.0],
])
graph = build_vecchia_graph(
    coordinates,
    covmodel=["exponentialC"],
    covparam=[(1.0, [0.7])],
    max_parents=2,
    ordering="input",
)

stream = StreamingSiteAccumulator(size=len(coordinates))
stream.update_many(
    node_indices=[0, 1, 0, 2],
    values=[0.9, 0.4, 1.1, 0.2],
    variances=[0.04, 0.05, 0.04, 0.06],
)
posterior_mean, factor = stream.posterior(graph.precision)
```

This is the preferred representation for a long history that repeatedly observes the same support. Observation variance must be positive because each term is a Gaussian likelihood.

## Sequential rank updates

`StreamingPrecisionState` factorizes a fixed sparse prior precision once. Each scalar observation adds a low-rank correction and updates the posterior mean without refactorizing the Vecchia graph.

```python
from stamps.bme import StreamingPrecisionState

state = StreamingPrecisionState(graph.precision)
diagnostic = state.update(node_index=0, value=0.9, variance=0.04)
state.update(node_index=2, value=0.2, variance=0.06)

posterior_mean = state.mean
posterior_variance = state.posterior_variance(indices=[0, 1, 2])
```

Report the innovation, innovation variance, standardized innovation, update count, and `graph_refactorized`. The retained low-rank basis grows with every update. Set an operational rebasing policy based on measured memory and latency; the class does not rebase automatically.

## Dense correctness reference

`StreamingRankState` updates a full covariance matrix. It is useful for tests and small examples, not for the large sparse production path.

```python
from stamps.bme import StreamingRankState

dense_state = StreamingRankState(
    mean=np.zeros(3),
    covariance=np.linalg.inv(graph.precision.toarray()),
)
dense_state.update(node_index=0, value=0.9, variance=0.04)
```

Use it to compare sparse streaming moments with an independent dense batch calculation.

## Operational boundaries

- Node indices refer to the fixed latent coordinate array used to build the precision matrix.
- A genuinely new coordinate changes the graph and is not a rank update.
- Aggregating observations is exact only for conditionally independent Gaussian likelihoods at the same latent node.
- Retain timestamps, source identifiers, and quality flags outside the sufficient statistics for auditability.
- A temporal state-transition or groundwater-flow model is separate from repeated likelihood updates. These utilities update a posterior on fixed support; they do not create a causal forecast model.
