# VISTA-BME user guide

## What this skill helps with

VISTA-BME combines exact or noisy measurements with probability distributions, intervals, censoring limits, and other likelihoods in one spatial or spatiotemporal estimate. The skill helps Codex translate a scientific data description into the correct VISTA-BME evidence objects, write and run Python, inspect approximation diagnostics, and explain the posterior.

Common requests include:

- "Use VISTA-BME to interpolate groundwater level from monitoring wells and three interval-valued estimates."
- "Represent detection limits and a skewed laboratory uncertainty distribution as soft data."
- "Check whether 16 or 32 Vecchia neighbors are sufficient compared with dense Gaussian BME."
- "Preserve the two modes in this Gaussian-mixture soft likelihood."
- "Update a fixed monitoring network as new observations arrive without rebuilding the graph."
- "Report RMSE, CRPS, interval coverage, computation time, and memory for an independent test set."

This skill does not choose a scientifically defensible covariance model from no information, turn arbitrary confidence scores into probability densities, or make a causal groundwater forecast without a transition model.

## Install from GitHub

In Codex, the simplest request is:

> Use `$skill-installer` to install the skill at `https://github.com/aar246860/vista-bme/tree/main/skills/vista-bme`.

Equivalent PowerShell command:

```powershell
python "$env:USERPROFILE\.codex\skills\.system\skill-installer\scripts\install-skill-from-github.py" `
  --repo aar246860/vista-bme `
  --path skills/vista-bme
```

Equivalent macOS or Linux command:

```bash
python ~/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo aar246860/vista-bme \
  --path skills/vista-bme
```

The skill becomes available on the next Codex turn. The repository is public and does not require repository-specific credentials.

## Install the numerical package

The Codex skill and the numerical package are separate. Install the package in the Python environment used for the analysis:

```powershell
python -m pip install "git+https://github.com/aar246860/vista-bme.git"
```

For development from a clone:

```powershell
git clone https://github.com/aar246860/vista-bme.git
Set-Location vista-bme
python -m pip install -e ".[dev]"
```

## Verify the installation

From a repository clone:

```powershell
python skills/vista-bme/scripts/smoke_test.py
python -m pytest -q test_vista_bme.py
```

The smoke test runs one non-Gaussian expectation-propagation estimate and one mixture-preserving estimate. It prints JSON containing posterior moments, convergence status, the variance method, and the retained mixture mass.

## Browser speed demonstration

Use the [VISTA BME Live Speed Lab](https://vista-bme-live.yflin.chatgpt.site) when a user wants to see a calculation respond immediately. The browser recomputes dense Gaussian BME and the exact ordered sparse calculation, then reports measured wall time, posterior RMSE, and matrix storage. Its source and tests are public at `https://github.com/aar246860/vista-bme-demo`.

The demonstration is deliberately limited to the one-dimensional exponential Gaussian case, where the first-order ordered representation is exact. Do not use its timing or equivalence result as proof for multidimensional graphs or non-Gaussian soft likelihoods; run the Python package and relevant diagnostics for those cases.

## First analysis request

Supply the following information when possible:

1. predicted quantity and units;
2. coordinate columns and their units;
3. target coordinates;
4. hard observation coordinates, values, and measurement variances;
5. soft observation coordinates and the distribution or bounds at each location;
6. covariance family and parameters, or data sufficient to estimate them;
7. whether the priority is dense equivalence, online speed, multimodal density preservation, or uncertainty calibration.

If information is missing, the skill should distinguish a harmless implementation choice from a scientific choice that requires the user or data owner.

## Expected result

A normal VISTA-BME handoff contains executable Python, posterior mean and marginal uncertainty, the selected numerical mode, and the diagnostics needed to interpret the approximation. Maps, time series, animations, and web demonstrations are downstream displays; they should be generated from stored posterior outputs rather than substituted for the numerical calculation.
