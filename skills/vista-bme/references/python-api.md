# Python API and recipes

## Core imports

```python
import numpy as np

from stamps.bme import (
    CensoredEvidence,
    GaussianEvidence,
    GaussianMixtureEvidence,
    IntervalEvidence,
    LogNormalEvidence,
    SkewNormalEvidence,
    StudentTEvidence,
    TabulatedEvidence,
    compile_vista_operator,
    gaussian_calibration_report,
    gaussian_exactness_report,
)
```

The target coordinates are `ck`, hard coordinates are `ch`, and soft coordinates are `cs`. Every coordinate array must have the same number and order of columns.

## Hard observations only

```python
targets = np.array([[0.25, 0.20], [0.75, 0.65]])
hard_coordinates = np.array([
    [0.00, 0.00],
    [1.00, 0.00],
    [0.50, 1.00],
])
hard_values = np.array([1.10, 0.45, 0.80])

operator = compile_vista_operator(
    ck=targets,
    ch=hard_coordinates,
    cs=np.empty((0, 2)),
    soft_evidence=[],
    covmodel=["exponentialC"],
    covparam=[(1.0, [0.60])],
    mode="vecchia",
    max_parents=16,
    ordering="random",
    random_state=7,
)
result = operator.predict(hard_values)

print(result.mean)
print(result.standard_deviation)
print(result.metadata)
```

An exact hard-data calculation uses `mode="exact"` and `max_parents=None`. Exact observations use the default `hard_variance=0.0`; pass a positive scalar or one variance per hard observation when measurement error is retained.

## Hard data with several soft likelihoods

```python
soft_coordinates = np.array([
    [0.20, 0.75],
    [0.80, 0.30],
    [0.55, 0.55],
])
soft_evidence = [
    GaussianEvidence(mean=0.70, variance=0.04),
    IntervalEvidence(lower=0.35, upper=0.65),
    CensoredEvidence(lower=-np.inf, upper=0.50),
]

operator = compile_vista_operator(
    ck=targets,
    ch=hard_coordinates,
    cs=soft_coordinates,
    soft_evidence=soft_evidence,
    covmodel=["exponentialC"],
    covparam=[(1.0, [0.60])],
    mode="vecchia",
    max_parents=16,
    ordering="random",
    random_state=7,
    ep_damping=0.5,
    ep_tolerance=1e-6,
    ep_max_iterations=50,
)
result = operator.predict(hard_values)

if not result.metadata["ep_converged"]:
    raise RuntimeError("Expectation propagation did not converge")
```

`predict()` accepts replacement evidence for the existing soft coordinates. The replacement list must remain aligned with `cs`.

```python
updated = operator.predict(
    hard_values=np.array([1.05, 0.48, 0.77]),
    soft_evidence=soft_evidence,
)
```

## Smooth and tabulated complex likelihoods

```python
grid = np.linspace(-0.5, 2.0, 301)
density = np.exp(-0.5 * ((grid - 0.75) / 0.18) ** 2)
density += 0.30 * np.exp(-0.5 * ((grid - 1.35) / 0.12) ** 2)

complex_evidence = [
    SkewNormalEvidence(location=0.65, scale=0.20, shape=5.0),
    LogNormalEvidence(log_mean=-0.30, log_scale=0.35),
    StudentTEvidence(location=0.80, scale=0.16, degrees_of_freedom=4.0),
    TabulatedEvidence(abscissa=grid, density=density),
]
```

Use these objects in the same position-by-position list supplied to `compile_vista_operator`. `predict()` approximates each non-Gaussian likelihood by a Gaussian expectation-propagation site and returns target moments. Inspect `moment_engines`, `maximum_relative_quadrature_error`, and the expectation-propagation residuals.

## Preserve a Gaussian-mixture posterior

```python
mixture_operator = compile_vista_operator(
    ck=np.array([[0.20, 0.00], [0.45, 0.00]]),
    ch=np.empty((0, 2)),
    cs=np.array([[0.00, 0.00]]),
    soft_evidence=[GaussianMixtureEvidence(
        weights=[0.45, 0.55],
        means=[-1.3, 1.4],
        variances=[0.05, 0.08],
    )],
    covmodel=["exponentialC"],
    covparam=[(1.0, [0.80])],
    mode="vecchia",
    max_parents=16,
    ordering="input",
)

mixture = mixture_operator.predict_mixture([], max_components=32)
x = np.linspace(-3.0, 3.0, 1001)
density_at_first_target = mixture.marginal_density(0, x)

print(mixture.weights)
print(mixture.component_mean[:, 0])
print(mixture.metadata["discarded_prior_mass"])
```

`predict_mixture()` currently accepts only `GaussianEvidence` and `GaussianMixtureEvidence`. The Cartesian product of components is truncated to `max_components`; report retained and discarded prior mass.

## Dense Gaussian equivalence

```python
common = dict(
    ck=targets,
    ch=hard_coordinates,
    cs=np.array([[0.50, 0.50]]),
    soft_evidence=[GaussianEvidence(mean=0.70, variance=0.04)],
    covmodel=["exponentialC"],
    covparam=[(1.0, [0.60])],
    max_parents=None,
)

dense = compile_vista_operator(mode="exact", **common).predict(hard_values)
full_graph = compile_vista_operator(
    mode="vecchia", ordering="input", **common
).predict(hard_values)

print(gaussian_exactness_report(dense, full_graph))
```

This comparison supports a Gaussian numerical-equivalence statement only. Non-Gaussian likelihood integration has its own approximation diagnostics.

## Calibration on independent observations

```python
report = gaussian_calibration_report(
    observed=test_values,
    mean=test_posterior_mean,
    variance=test_posterior_variance,
    levels=(0.90, 0.95),
)
```

Use spatially separated wells, future time periods, or another leakage-safe design. The report includes proper scores, interval coverage and width, and probability-integral-transform summaries.
