# Data and soft-evidence contract

## Coordinate arrays

- `ck`: target coordinates, shape `(n_targets, dimensions)`.
- `ch`: hard-observation coordinates, shape `(n_hard, dimensions)`.
- `cs`: soft-observation coordinates, shape `(n_soft, dimensions)`.
- `hard_values`: one finite value per row of `ch`.
- `hard_variance`: zero for values treated as exact at the analysis scale, or a positive scalar or vector for noisy measurements.
- `soft_evidence`: one evidence object per row of `cs`, in the same order.

Use the same coordinate columns and units in all three arrays. Separate identifiers and provenance from the numerical coordinate matrix. Remove or explicitly model missing values before creating arrays.

For spatiotemporal work, a time column is a covariance coordinate, not merely a label. Its numerical scale and covariance parameters must represent the intended temporal dependence. Do not append Unix timestamps to spatial coordinates without specifying the corresponding covariance model and units.

## Evidence classes

| Scientific information | Python object | Parameters and meaning |
|---|---|---|
| Noisy estimate with symmetric Gaussian error | `GaussianEvidence` | `mean`; error `variance`, not standard deviation |
| Value known to lie in a finite range | `IntervalEvidence` | `lower < upper` |
| Below a detection limit | `CensoredEvidence` | `lower=-np.inf`, finite `upper` |
| Above a reporting threshold | `CensoredEvidence` | finite `lower`, `upper=np.inf` |
| Asymmetric smooth likelihood | `SkewNormalEvidence` | `location`, positive `scale`, `shape` |
| Positive likelihood with log-scale parameters | `LogNormalEvidence` | `log_mean`, positive `log_scale` |
| Heavy-tailed symmetric likelihood | `StudentTEvidence` | `location`, positive `scale`, degrees of freedom greater than 2 |
| A small number of distinct alternatives | `GaussianMixtureEvidence` | nonnegative `weights`, `means`, positive component `variances` |
| Empirical or externally calculated density | `TabulatedEvidence` | strictly increasing `abscissa` and nonnegative `density` with positive area |

Weights and tabulated densities are normalized by the evidence objects. Tabulated likelihoods use linear interpolation between supplied values and zero density outside the tabulated support.

## Do not manufacture a soft PDF

A range, quantile pair, ensemble, model output, detection limit, and expert judgment carry different information. Convert them as follows only when the stated measurement process supports the choice:

- Use an interval when only membership inside bounds is defensible.
- Fit a distribution to an ensemble only after checking member weighting, dependence, and whether the ensemble represents observational uncertainty or process variability.
- Use a tabulated likelihood when the source provides a density or when a documented transformation yields one.
- Keep source reliability and measurement error separate from the field covariance when possible.
- Preserve the original units and transformation. If the field is modeled on a logarithmic scale, transform hard values, soft supports, targets, and interpretation consistently.

## Recommended exchange tables

VISTA-BME does not require a particular file format. For a reproducible adapter, use:

`hard.csv`

```text
id,x,y,time,value,variance,unit,source
```

`targets.csv`

```text
id,x,y,time
```

`soft.jsonl`

```json
{"id":"s1","coordinates":[0.2,0.7],"type":"gaussian","mean":0.8,"variance":0.04,"unit":"m"}
{"id":"s2","coordinates":[0.8,0.3],"type":"interval","lower":0.4,"upper":0.7,"unit":"m"}
{"id":"s3","coordinates":[0.5,0.5],"type":"censored","lower":null,"upper":0.5,"unit":"m"}
{"id":"s4","coordinates":[0.3,0.4],"type":"gaussian_mixture","weights":[0.4,0.6],"means":[-1.0,1.2],"variances":[0.05,0.08],"unit":"m"}
```

Treat JSON `null` as negative or positive infinity only according to the declared censoring side. Record the covariance specification, coordinate reference system, time zone, transformations, quality filters, and source retrieval date in a separate analysis manifest.

## Covariance inputs

`covmodel` and `covparam` use STAMPS covariance syntax. A basic two-dimensional example is:

```python
covmodel = ["exponentialC"]
covparam = [(1.0, [0.60])]
```

The sill and range must use the modeled variable and coordinate units. Fit or justify them from training data only when evaluating independent predictions. Never tune covariance parameters on held-out targets and then report those targets as independent validation.
