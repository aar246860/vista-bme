# VISTA-BME

VISTA-BME is a Python implementation of **Vecchia Inference for Streaming Temporal Assimilation in Bayesian Maximum Entropy**. It retains the dense Gaussian BME calculation as a numerical reference, replaces the dense prior with an ordered sparse Vecchia graph for larger problems, and assimilates non-Gaussian soft evidence with expectation propagation or bounded mixture propagation.

The repository contains only the numerical package, tests, and an installable Codex skill. Research data, manuscript builds, and large benchmark outputs are maintained separately.

## Live speed demonstration

The [VISTA BME Live Speed Lab](https://vista-bme-live.yflin.chatgpt.site) recomputes dense Gaussian BME and the exact ordered sparse path in the visitor's browser. It reports measured wall time, posterior disagreement, and matrix storage as the workload changes. Its public source and validation tests are in [`aar246860/vista-bme-demo`](https://github.com/aar246860/vista-bme-demo).

The browser lab uses the one-dimensional exponential Gaussian case, where a first-order ordered representation is exact. It does not substitute for the Python package's multidimensional non-Gaussian inference.

## What it supports

- exact Gaussian BME posterior moments for a fixed covariance model;
- sparse Vecchia inference with a configurable predecessor count;
- hard observations with exact or noisy measurements;
- Gaussian, interval, censored, skew-normal, lognormal, Student-t, Gaussian-mixture, and tabulated soft likelihoods;
- expectation-propagation convergence and quadrature diagnostics;
- mixture-preserving inference for manageable Gaussian mixtures;
- fixed-support streaming updates;
- exactness and uncertainty-calibration reports.

Full-predecessor numerical equivalence applies to the Gaussian case. Reducing the predecessor count introduces a sparse prior approximation. Non-Gaussian expectation propagation introduces a separate likelihood approximation; the package reports diagnostics for both rather than treating them as exact BME.

## Install the Python package

```bash
python -m pip install "git+https://github.com/aar246860/vista-bme.git"
```

For development:

```bash
git clone https://github.com/aar246860/vista-bme.git
cd vista-bme
python -m pip install -e ".[dev]"
python -m pytest
```

## First calculation

```python
import numpy as np
from vista_bme import GaussianEvidence, compile_vista_operator

targets = np.array([[0.25], [0.50], [0.75]])
hard_locations = np.array([[0.00], [1.00]])
hard_values = np.array([0.2, 1.1])
soft_locations = np.array([[0.40], [0.80]])
soft = (
    GaussianEvidence(mean=0.55, variance=0.04),
    GaussianEvidence(mean=0.90, variance=0.09),
)

operator = compile_vista_operator(
    targets,
    ch=hard_locations,
    cs=soft_locations,
    soft_evidence=soft,
    covmodel=["exponentialC"],
    covparam=[(1.0, [0.30])],
    mode="vecchia",
    max_parents=16,
    ordering="lexicographic",
)
result = operator.predict(hard_values)

print(result.mean)
print(result.standard_deviation)
print(result.metadata)
```

See the skill references for complex evidence, dense equivalence checks, calibration, and streaming examples:

- [`python-api.md`](skills/vista-bme/references/python-api.md)
- [`evidence-and-data.md`](skills/vista-bme/references/evidence-and-data.md)
- [`diagnostics.md`](skills/vista-bme/references/diagnostics.md)
- [`streaming.md`](skills/vista-bme/references/streaming.md)

## Install the Codex skill

Ask Codex:

> Use `$skill-installer` to install the skill at `https://github.com/aar246860/vista-bme/tree/main/skills/vista-bme`.

Or run:

```bash
python ~/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo aar246860/vista-bme \
  --path skills/vista-bme
```

After installation, requests can invoke the skill explicitly:

> Use `$vista-bme` to interpolate these hard observations and interval-valued soft data. Compare 16 and 32 predecessors, then report posterior uncertainty, EP convergence, and sparse sensitivity.

The numerical package and Codex skill are separate installations. Install the package in the Python environment that will execute the analysis.

## Verify an installation

```bash
python skills/vista-bme/scripts/smoke_test.py
python -m pytest
```

The smoke test exercises non-Gaussian expectation propagation and mixture-preserving inference. It prints convergence, quadrature, variance, and retained-mixture diagnostics as JSON.

## License and attribution

VISTA-BME extends numerical components from STAMPS and is distributed under the GNU General Public License v3. See [`LICENSE`](LICENSE) and [`CITATION.cff`](CITATION.cff).
