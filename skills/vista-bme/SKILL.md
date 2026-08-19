---
name: vista-bme
description: Formulate, run, validate, and interpret VISTA-BME spatial or spatiotemporal estimates that combine hard observations with distribution-valued soft evidence. Use for Vecchia sparse BME, complex or multimodal soft likelihoods, fixed-support streaming updates, posterior calibration, or comparison with dense Gaussian BME; do not use for generic kriging or machine learning when VISTA-BME and observational uncertainty are not central.
metadata:
  short-description: Sparse BME with complex soft evidence
---

# VISTA-BME

Use the `vista-bme` Python package to produce an executable analysis and an evidence-bounded interpretation. VISTA-BME means Vecchia Inference for Streaming Temporal Assimilation in Bayesian Maximum Entropy.

## Route the request

- For installation, a first run, or example prompts, read [references/user-guide.md](references/user-guide.md).
- For executable Python, API arguments, and prediction examples, read [references/python-api.md](references/python-api.md).
- For coordinate arrays, units, hard observations, and selection of a soft-evidence class, read [references/evidence-and-data.md](references/evidence-and-data.md).
- For exactness, sparse sensitivity, expectation-propagation diagnostics, uncertainty calibration, or troubleshooting, read [references/diagnostics.md](references/diagnostics.md).
- For incoming observations at fixed latent nodes, read [references/streaming.md](references/streaming.md).

Read only the references needed for the current request. When several modes are requested, read them in the order listed above.

## Required workflow

1. Establish the prediction quantity, its units, coordinate dimensions, covariance model, target coordinates, hard observations, and the meaning of every soft likelihood. Do not infer an uncertainty distribution from a label such as "soft" alone.
2. Confirm that `stamps.bme.compile_vista_operator` and all requested evidence classes import. If the package is absent, explain the installation step from the user guide before attempting calculations; do not install network dependencies without authorization.
3. Validate array lengths, finite values, coordinate order, units, positive variances, likelihood support, and alignment between `cs` and `soft_evidence`.
4. Choose the calculation from the table below. Compile the spatial operator once for a fixed coordinate graph and covariance model, then reuse it for value updates.
5. Return posterior mean and marginal variance or standard deviation together with the relevant metadata. For non-Gaussian evidence, always report convergence and approximation diagnostics.
6. When accuracy claims matter, run the applicable dense-limit, parent-count, calibration, or mixture-retention comparison before drawing a conclusion.

| Need | Calculation |
|---|---|
| Dense Gaussian BME moments | `mode="exact"`, Gaussian soft evidence, `max_parents=None` |
| Sparse Gaussian or non-Gaussian BME | `mode="vecchia"`, a stated `max_parents` |
| Gaussian full-predecessor equivalence gate | compare exact mode with Vecchia mode using `max_parents=None` |
| Scalable moments for smooth, bounded, censored, skewed, or heavy-tailed evidence | `predict()`; non-Gaussian likelihoods use expectation propagation |
| Distinct modes from a small Gaussian mixture | `predict_mixture()` with an explicit component budget |
| Repeated Gaussian observations at fixed nodes | `StreamingSiteAccumulator` or `StreamingPrecisionState` |

## Scientific boundaries

- Numerical equivalence to dense BME applies to the Gaussian case when all predecessors are retained and the same trend, covariance, coordinates, and observation variances are used.
- Reduced `max_parents` introduces a sparse-prior approximation. Expectation propagation introduces a separate likelihood approximation. Report them separately.
- `predict()` returns marginal moments. It does not preserve a multimodal target density. Use `predict_mixture()` when retained modes are scientifically important and the component count is manageable.
- Vecchia mode currently uses a zero residual trend. Estimate or remove a deterministic mean before fitting the residual field when a nonzero mean structure is needed.
- Marginal variance switches to a stochastic Hutchinson estimate above `variance_exact_limit`. Report `variance_method` and assess Monte Carlo stability when uncertainty maps are central.
- Streaming rank updates assume fixed latent support. A new coordinate requires rebuilding or extending the graph. `StreamingPrecisionState` retains a low-rank history whose size grows with updates and therefore needs periodic rebasing in a long-running service.
- One million incoming observations aggregated onto fixed nodes is not one million latent spatial nodes. Keep these scale claims distinct.
- Do not present the browser demonstration's local "BME-style" interpolation as the full Python VISTA-BME calculation unless it actually calls this package with the stated model.

## Minimum handoff

Provide:

- the executable code or command used;
- shapes and units for targets, hard data, and soft data;
- covariance model, parameters, ordering, parent count, random seed, and variance method;
- evidence families and their parameterization;
- posterior mean and uncertainty at requested targets;
- expectation-propagation or mixture diagnostics when applicable;
- validation design and metrics when comparative claims are made;
- limitations tied to the executed calculation, not generic disclaimers.

Run `python scripts/smoke_test.py` from this skill directory after installing the package when a quick environment check is useful.
