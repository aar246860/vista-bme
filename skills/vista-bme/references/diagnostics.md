# Diagnostics, validation, and troubleshooting

## Read the result metadata

For `predict()`, report the keys relevant to the executed mode:

- `mode`: exact Gaussian, sparse Gaussian, or Vecchia expectation propagation.
- `exact_limit`: whether every predecessor was retained in the Vecchia graph.
- `max_parents` and `graph_nonzeros`: sparse-graph size.
- `variance_method`: exact selected inverse or a Hutchinson estimate and sample count.
- `ep_converged`, `ep_iterations`, and `ep_max_site_change`: fixed-point behavior.
- `ep_standardized_mean_residual` and `ep_relative_variance_residual`: agreement between the returned marginal and the tilted moments at soft sites.
- `ep_precision_clips`: stabilization events that should be disclosed and investigated.
- `moment_engines` and `maximum_relative_quadrature_error`: how non-Gaussian tilted moments were evaluated and their numerical convergence estimate.
- `factorizations` and `ep_reused_converged_factor`: computational work in the likelihood update.

For `predict_mixture()`, also report:

- `component_combinations` and `retained_components`;
- `retained_prior_mass` and `discarded_prior_mass`;
- `effective_posterior_components`.

Do not reduce these fields to a single "passed" label when the numeric values affect interpretation.

## Four separate validation questions

### 1. Gaussian representation

Compare dense exact mode with Vecchia mode using `max_parents=None`, identical coordinate order, covariance, trend, hard variance, and Gaussian evidence. Use `gaussian_exactness_report`. This isolates numerical representation.

### 2. Sparse-prior approximation

Repeat predictions for several parent counts, for example 8, 16, 32, and 64, while holding ordering and random seed fixed. Compare posterior means, variances, target scores, time, and memory with a feasible dense or larger-parent calculation. Parent-count error is not expectation-propagation error.

### 3. Likelihood approximation

For bounded, censored, skewed, heavy-tailed, or tabulated likelihoods, inspect convergence, moment residuals, quadrature error, and precision clipping. For representative low-dimensional cases, compare target densities or moments with numerical integration. For strong multimodality, compare `predict()` with `predict_mixture()` and examine modes, tail probabilities, and discarded component mass.

### 4. Field prediction and calibration

Use independent wells, spatial blocks, future periods, or entire regions. Report point error and probabilistic behavior separately:

- MAE and RMSE;
- CRPS or another proper score;
- interval coverage and mean interval width together;
- negative log predictive density when variance is reliable;
- Brier score for scientifically defined thresholds;
- wall time, update latency, and peak memory.

Estimate covariance, deterministic trends, machine-learning features, calibration factors, and stopping rules using training data only.

## Variance at large node counts

When the free graph exceeds `variance_exact_limit`, VISTA-BME estimates the inverse diagonal with `variance_samples` Hutchinson probes. If an uncertainty map is central to the conclusion:

1. repeat the calculation with more probes and at least one additional random seed;
2. compare variance at selected nodes with exact selected inversion on a smaller graph;
3. store the stochastic standard error or replicate variation;
4. avoid interpreting small contour differences that are below this numerical variation.

## Troubleshooting

| Symptom | Likely cause | Response |
|---|---|---|
| `soft_evidence must match` | `cs` rows and evidence list differ | rebuild both from one keyed table and verify order |
| exact mode rejects evidence | a likelihood is not Gaussian | use Vecchia mode or create a separate low-dimensional numerical reference |
| Vecchia mode rejects trend | nonzero trend supplied | fit/remove the deterministic mean and model residuals, or use supported exact Gaussian mode |
| expectation propagation does not converge | difficult likelihood interaction or aggressive updates | inspect units and supports, reduce damping, increase iterations, and compare a representative case with integration |
| large moment residual | one Gaussian site is inadequate | inspect the density; use mixture preservation for a manageable Gaussian mixture |
| precision clips occur | extremely narrow likelihood or poor scaling | check units, transformation, interval width, and covariance variance before changing safeguards |
| noisy uncertainty surface | too few Hutchinson probes | increase `variance_samples`, repeat seeds, or compute exact selected variances for critical nodes |
| mixture mass discarded | component product exceeds budget | increase the budget if feasible, simplify negligible components with documentation, or report the lost mass |
| streaming memory grows | rank history is long | aggregate repeated nodes or periodically rebuild the posterior base state |

## Claim language

Use "numerically equivalent to dense Gaussian BME" only after the full-predecessor comparison. Use "close to dense BME" with the measured parent-count error. Use "expectation-propagation approximation" for non-Gaussian moments. Use "mixture-preserving calculation" only when the retained-component diagnostics are reported. State the tested observation count separately from the number of unique latent nodes.
