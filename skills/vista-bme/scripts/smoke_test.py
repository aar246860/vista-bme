"""Run a small VISTA-BME environment and inference smoke test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Permit a smoke test directly from a source checkout. An installed skill does
# not contain the package, so the normal package-installation message remains
# active outside the repository.
repository_root = Path(__file__).resolve().parents[3]
if (repository_root / "stamps" / "bme").is_dir():
    sys.path.insert(0, str(repository_root))

try:
    from stamps.bme import (
        GaussianMixtureEvidence,
        IntervalEvidence,
        StudentTEvidence,
        compile_vista_operator,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "vista-bme is not installed. Install the numerical package before "
        "running this smoke test. See references/user-guide.md."
    ) from exc


def main() -> None:
    targets = np.array([[0.25, 0.20], [0.75, 0.65]])
    hard_coordinates = np.array([
        [0.00, 0.00],
        [1.00, 0.00],
        [0.50, 1.00],
    ])
    hard_values = np.array([1.10, 0.45, 0.80])
    soft_coordinates = np.array([[0.20, 0.75], [0.80, 0.30]])

    operator = compile_vista_operator(
        ck=targets,
        ch=hard_coordinates,
        cs=soft_coordinates,
        soft_evidence=[
            IntervalEvidence(0.55, 0.90),
            StudentTEvidence(
                location=0.55,
                scale=0.18,
                degrees_of_freedom=4.0,
            ),
        ],
        covmodel=["exponentialC"],
        covparam=[(1.0, [0.60])],
        mode="vecchia",
        max_parents=4,
        ordering="input",
        ep_damping=0.5,
        ep_tolerance=1e-7,
        ep_max_iterations=80,
    )
    result = operator.predict(hard_values)
    if not result.metadata["ep_converged"]:
        raise RuntimeError("Expectation propagation did not converge in the smoke test.")
    if not (np.isfinite(result.mean).all() and np.isfinite(result.variance).all()):
        raise RuntimeError("The smoke test returned non-finite posterior moments.")

    mixture_operator = compile_vista_operator(
        ck=np.array([[0.15, 0.00]]),
        ch=np.empty((0, 2)),
        cs=np.array([[0.00, 0.00]]),
        soft_evidence=[GaussianMixtureEvidence(
            weights=[0.45, 0.55],
            means=[-1.25, 1.30],
            variances=[0.05, 0.08],
        )],
        covmodel=["exponentialC"],
        covparam=[(1.0, [0.80])],
        mode="vecchia",
        max_parents=None,
        ordering="input",
    )
    mixture = mixture_operator.predict_mixture([], max_components=8)
    if not np.isclose(mixture.weights.sum(), 1.0):
        raise RuntimeError("Mixture posterior weights do not sum to one.")

    payload = {
        "status": "ok",
        "posterior_mean": result.mean.tolist(),
        "posterior_standard_deviation": result.standard_deviation.tolist(),
        "ep_converged": result.metadata["ep_converged"],
        "ep_iterations": result.metadata["ep_iterations"],
        "variance_method": result.metadata["variance_method"],
        "maximum_relative_quadrature_error": result.metadata[
            "maximum_relative_quadrature_error"
        ],
        "mixture_components": mixture.metadata["retained_components"],
        "mixture_retained_prior_mass": mixture.metadata["retained_prior_mass"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
