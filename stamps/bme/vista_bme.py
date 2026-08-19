"""VISTA-BME: sparse and streaming inference for Bayesian Maximum Entropy.

VISTA-BME (Vecchia Inference for Streaming Temporal Assimilation in BME)
preserves the existing compiled STAMPS Gaussian BME operator as its exact
limit.  A Vecchia graph supplies a sparse Gaussian prior for large problems,
and damped expectation propagation (EP) converts interval-valued soft evidence
into Gaussian site terms.

This module is an intentionally small numerical contract for the research
method.  It does not yet claim production-scale graph construction or sparse
Cholesky rank updates; those capabilities are benchmarked and gated separately.
"""

from __future__ import division

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import solve_triangular
from scipy.integrate import quad, simpson, trapezoid
from scipy.special import logsumexp
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from scipy.stats import lognorm, norm, skewnorm, t as student_t, truncnorm

from stamps.general.coord2K import coord2K

from .fast_bme import compile_bme_operator


@dataclass(frozen=True)
class GaussianEvidence:
    """Gaussian soft evidence represented by a mean and error variance."""

    mean: float
    variance: float

    def __post_init__(self):
        if not np.isfinite(self.mean):
            raise ValueError("Gaussian evidence mean must be finite.")
        if not np.isfinite(self.variance) or self.variance <= 0:
            raise ValueError("Gaussian evidence variance must be positive.")


@dataclass(frozen=True)
class IntervalEvidence:
    """Soft evidence that constrains a latent value to an interval."""

    lower: float
    upper: float

    def __post_init__(self):
        if np.isnan(self.lower) or np.isnan(self.upper):
            raise ValueError("Interval bounds cannot be NaN.")
        if self.lower >= self.upper:
            raise ValueError("Interval lower bound must be less than upper bound.")


@dataclass(frozen=True)
class CensoredEvidence:
    """One- or two-sided censoring likelihood for a latent value."""

    lower: float = -np.inf
    upper: float = np.inf

    def __post_init__(self):
        if np.isnan(self.lower) or np.isnan(self.upper):
            raise ValueError("Censoring bounds cannot be NaN.")
        if self.lower >= self.upper:
            raise ValueError("Censoring lower bound must be less than upper bound.")
        if not np.isfinite(self.lower) and not np.isfinite(self.upper):
            raise ValueError("At least one censoring bound must be finite.")


@dataclass(frozen=True)
class SkewNormalEvidence:
    """Skew-normal soft likelihood."""

    location: float
    scale: float
    shape: float

    def __post_init__(self):
        if not np.isfinite((self.location, self.scale, self.shape)).all():
            raise ValueError("Skew-normal parameters must be finite.")
        if self.scale <= 0:
            raise ValueError("Skew-normal scale must be positive.")


@dataclass(frozen=True)
class LogNormalEvidence:
    """Lognormal soft likelihood parameterized on the logarithmic scale."""

    log_mean: float
    log_scale: float

    def __post_init__(self):
        if not np.isfinite((self.log_mean, self.log_scale)).all():
            raise ValueError("Lognormal parameters must be finite.")
        if self.log_scale <= 0:
            raise ValueError("Lognormal log_scale must be positive.")


@dataclass(frozen=True)
class StudentTEvidence:
    """Student-t soft likelihood with finite mean and variance."""

    location: float
    scale: float
    degrees_of_freedom: float

    def __post_init__(self):
        if not np.isfinite(
                (self.location, self.scale, self.degrees_of_freedom)).all():
            raise ValueError("Student-t parameters must be finite.")
        if self.scale <= 0 or self.degrees_of_freedom <= 2:
            raise ValueError(
                "Student-t scale must be positive and degrees_of_freedom must exceed 2.")


@dataclass(frozen=True)
class GaussianMixtureEvidence:
    """Normalized finite Gaussian-mixture soft likelihood."""

    weights: Sequence[float]
    means: Sequence[float]
    variances: Sequence[float]

    def __post_init__(self):
        weights = np.asarray(self.weights, dtype=float).reshape(-1)
        means = np.asarray(self.means, dtype=float).reshape(-1)
        variances = np.asarray(self.variances, dtype=float).reshape(-1)
        if weights.size == 0 or not (weights.size == means.size == variances.size):
            raise ValueError("Mixture weights, means, and variances must have equal nonzero length.")
        if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Mixture weights must be finite, non-negative, and have positive sum.")
        if not np.isfinite(means).all():
            raise ValueError("Mixture means must be finite.")
        if not np.isfinite(variances).all() or np.any(variances <= 0):
            raise ValueError("Mixture variances must be finite and positive.")
        object.__setattr__(self, "weights", tuple((weights / weights.sum()).tolist()))
        object.__setattr__(self, "means", tuple(means.tolist()))
        object.__setattr__(self, "variances", tuple(variances.tolist()))


@dataclass(frozen=True)
class TabulatedEvidence:
    """Soft likelihood supplied as ordered abscissa and non-negative density."""

    abscissa: Sequence[float]
    density: Sequence[float]

    def __post_init__(self):
        abscissa = np.asarray(self.abscissa, dtype=float).reshape(-1)
        density = np.asarray(self.density, dtype=float).reshape(-1)
        if abscissa.size < 3 or abscissa.size != density.size:
            raise ValueError("Tabulated evidence requires at least three paired values.")
        if not np.isfinite(abscissa).all() or np.any(np.diff(abscissa) <= 0):
            raise ValueError("Tabulated abscissa must be finite and strictly increasing.")
        if not np.isfinite(density).all() or np.any(density < 0):
            raise ValueError("Tabulated density must be finite and non-negative.")
        area = float(trapezoid(density, abscissa))
        if not np.isfinite(area) or area <= 0:
            raise ValueError("Tabulated density must have positive finite integral.")
        object.__setattr__(self, "abscissa", tuple(abscissa.tolist()))
        object.__setattr__(self, "density", tuple((density / area).tolist()))


SoftEvidence = Union[
    GaussianEvidence,
    IntervalEvidence,
    CensoredEvidence,
    SkewNormalEvidence,
    LogNormalEvidence,
    StudentTEvidence,
    GaussianMixtureEvidence,
    TabulatedEvidence,
]
SOFT_EVIDENCE_TYPES = (
    GaussianEvidence,
    IntervalEvidence,
    CensoredEvidence,
    SkewNormalEvidence,
    LogNormalEvidence,
    StudentTEvidence,
    GaussianMixtureEvidence,
    TabulatedEvidence,
)
_GL_NODES_256, _GL_WEIGHTS_256 = np.polynomial.legendre.leggauss(256)
_GL_NODES_512, _GL_WEIGHTS_512 = np.polynomial.legendre.leggauss(512)


@dataclass
class VecchiaGraph:
    """Sparse Gaussian precision induced by ordered Vecchia conditionals.

    ``precision`` is returned in the caller's original coordinate order.
    ``parents[i]`` contains original coordinate indices for node ``i`` in
    ordered-factor position, and ``coefficients[i]`` contains its conditional
    regression coefficients in matching order.
    """

    precision: csr_matrix
    order: np.ndarray
    inverse_order: np.ndarray
    parents: Tuple[np.ndarray, ...]
    coefficients: Tuple[np.ndarray, ...]
    conditional_variance: np.ndarray
    metadata: dict = field(default_factory=dict)


@dataclass
class VISTAResult:
    """Posterior marginal moments and inference diagnostics."""

    mean: np.ndarray
    variance: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def standard_deviation(self):
        return np.sqrt(np.maximum(self.variance, 0.0))


@dataclass
class VISTAMixtureResult(VISTAResult):
    """Mixture-preserving posterior marginals for Gaussian-mixture evidence."""

    weights: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    component_mean: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float))
    component_variance: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float))

    def marginal_density(self, target_index, values):
        """Evaluate one target marginal density on ``values``."""
        target_index = int(target_index)
        values = np.asarray(values, dtype=float)
        if target_index < 0 or target_index >= self.mean.size:
            raise IndexError("target_index is outside the prediction targets.")
        density = np.zeros_like(values, dtype=float)
        for weight, mean, variance in zip(
                self.weights,
                self.component_mean[:, target_index],
                self.component_variance[:, target_index]):
            density += weight * norm.pdf(values, loc=mean, scale=np.sqrt(variance))
        return density


def _as_coordinates(value, name, dimensions=None, allow_empty=False):
    if value is None:
        if dimensions is None:
            raise ValueError("dimensions are required for an omitted coordinate set")
        return np.empty((0, dimensions), dtype=float)
    result = np.asarray(value, dtype=float)
    if result.ndim == 1:
        result = result.reshape(1, -1)
    if result.ndim != 2:
        raise ValueError("%s must be a two-dimensional coordinate array." % name)
    if dimensions is not None and result.shape[1] != dimensions:
        raise ValueError("%s must have %d coordinate columns." % (name, dimensions))
    if not allow_empty and result.shape[0] == 0:
        raise ValueError("%s cannot be empty." % name)
    if not np.all(np.isfinite(result)):
        raise ValueError("%s contains non-finite coordinates." % name)
    return result


def _as_values(value, size, name):
    if value is None and size == 0:
        return np.empty(0, dtype=float)
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != size:
        raise ValueError("%s must contain %d values." % (name, size))
    if not np.all(np.isfinite(result)):
        raise ValueError("%s contains non-finite values." % name)
    return result


def _broadcast_variance(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        result = np.full(size, float(result), dtype=float)
    else:
        result = result.reshape(-1)
    if result.size != size:
        raise ValueError("%s must be scalar or contain %d values." % (name, size))
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("%s must be finite and non-negative." % name)
    return result


def _covariance(left, right, covmodel, covparam):
    return np.asarray(coord2K(left, right, covmodel, covparam)[0], dtype=float)


def _resolve_order(coordinates, ordering, random_state):
    count = coordinates.shape[0]
    if isinstance(ordering, str):
        key = ordering.strip().lower()
        if key == "input":
            return np.arange(count, dtype=np.int64)
        if key == "random":
            return np.random.default_rng(random_state).permutation(count).astype(np.int64)
        if key in ("lexicographic", "space_time"):
            keys = tuple(coordinates[:, column] for column in range(coordinates.shape[1] - 1, -1, -1))
            return np.lexsort(keys).astype(np.int64)
        raise ValueError("ordering must be 'input', 'random', 'lexicographic', or a permutation.")
    order = np.asarray(ordering, dtype=np.int64).reshape(-1)
    if order.size != count or not np.array_equal(np.sort(order), np.arange(count)):
        raise ValueError("ordering must be a permutation of all coordinate indices.")
    return order


def _is_zero_trend(trend):
    if trend is None:
        return True
    if isinstance(trend, str):
        return trend.strip().lower() == "zero"
    try:
        return bool(np.isscalar(trend) and np.isnan(trend))
    except TypeError:
        return False


def build_vecchia_graph(
        coordinates, covmodel, covparam, max_parents=32,
        ordering="random", random_state=0, workers=1,
        jitter=1e-12, candidate_multiplier=8):
    """Build a Vecchia conditional graph and sparse prior precision.

    Setting ``max_parents=None`` retains every predecessor.  In this limit the
    conditional factorization reconstructs the dense Gaussian covariance up to
    floating-point tolerance and is used as the exactness gate.
    """
    coordinates = _as_coordinates(coordinates, "coordinates")
    count = coordinates.shape[0]
    if max_parents is not None:
        if not isinstance(max_parents, (int, np.integer)) or max_parents < 1:
            raise ValueError("max_parents must be a positive integer or None.")
        max_parents = int(max_parents)
    if not isinstance(workers, (int, np.integer)) or workers == 0 or workers < -1:
        raise ValueError("workers must be -1 or a positive integer.")
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError("jitter must be finite and non-negative.")

    order = _resolve_order(coordinates, ordering, random_state)
    inverse_order = np.empty(count, dtype=np.int64)
    inverse_order[order] = np.arange(count, dtype=np.int64)
    ordered = coordinates[order]
    coordinate_scale = np.std(ordered, axis=0)
    coordinate_scale[coordinate_scale <= np.finfo(float).eps] = 1.0

    neighbour_candidates = None
    if max_parents is not None and max_parents < count - 1:
        search_coordinates = ordered / coordinate_scale
        candidate_count = min(
            count,
            max(64, max_parents * max(int(candidate_multiplier), 2) + 1),
        )
        tree = cKDTree(search_coordinates)
        _, neighbour_candidates = tree.query(
            search_coordinates, k=candidate_count, workers=int(workers))
        neighbour_candidates = np.asarray(neighbour_candidates, dtype=np.int64)
        if neighbour_candidates.ndim == 1:
            neighbour_candidates = neighbour_candidates[:, None]

    row_indices = []
    column_indices = []
    values = []
    parents_ordered = []
    coefficients = []
    conditional_variance = np.empty(count, dtype=float)
    stabilised_nodes = 0

    for node in range(count):
        if node == 0:
            parent = np.empty(0, dtype=np.int64)
        elif max_parents is None or max_parents >= node:
            parent = np.arange(node, dtype=np.int64)
        else:
            candidates = neighbour_candidates[node]
            candidates = candidates[candidates < node]
            candidates = np.unique(candidates)
            if candidates.size < max_parents:
                delta = ordered[:node] - ordered[node]
                distances = np.sum((delta / coordinate_scale) ** 2, axis=1)
                parent = np.argsort(distances)[:max_parents].astype(np.int64)
            else:
                delta = ordered[candidates] - ordered[node]
                distance = np.sum((delta / coordinate_scale) ** 2, axis=1)
                parent = candidates[np.argsort(distance)[:max_parents]]
        parent = np.asarray(parent, dtype=np.int64)

        diagonal = float(_covariance(
            ordered[node:node + 1], ordered[node:node + 1],
            covmodel, covparam)[0, 0])
        if diagonal <= 0 or not np.isfinite(diagonal):
            raise np.linalg.LinAlgError("Covariance diagonal must be positive.")
        if parent.size:
            parent_coordinates = ordered[parent]
            parent_covariance = _covariance(
                parent_coordinates, parent_coordinates, covmodel, covparam)
            cross_covariance = _covariance(
                ordered[node:node + 1], parent_coordinates,
                covmodel, covparam).reshape(-1)
            try:
                coefficient = np.linalg.solve(parent_covariance, cross_covariance)
            except np.linalg.LinAlgError:
                stabilised_nodes += 1
                regularised = parent_covariance.copy()
                regularised[np.diag_indices_from(regularised)] += jitter * diagonal
                try:
                    coefficient = np.linalg.solve(regularised, cross_covariance)
                except np.linalg.LinAlgError:
                    coefficient = np.linalg.pinv(regularised) @ cross_covariance
            innovation_variance = diagonal - float(np.dot(cross_covariance, coefficient))
        else:
            coefficient = np.empty(0, dtype=float)
            innovation_variance = diagonal
        minimum_variance = max(jitter * diagonal, np.finfo(float).eps * diagonal)
        if not np.isfinite(innovation_variance) or innovation_variance <= 0:
            innovation_variance = minimum_variance
        innovation_variance = max(innovation_variance, minimum_variance)

        row_indices.append(node)
        column_indices.append(node)
        values.append(1.0)
        for parent_index, coefficient_value in zip(parent, coefficient):
            row_indices.append(node)
            column_indices.append(int(parent_index))
            values.append(-float(coefficient_value))
        parents_ordered.append(parent)
        coefficients.append(np.asarray(coefficient, dtype=float))
        conditional_variance[node] = innovation_variance

    lower = coo_matrix(
        (values, (row_indices, column_indices)), shape=(count, count)).tocsr()
    ordered_precision = (
        lower.T @ diags(1.0 / conditional_variance) @ lower).tocsr()
    precision = ordered_precision[inverse_order, :][:, inverse_order].tocsr()
    original_parents = tuple(order[parent] for parent in parents_ordered)

    return VecchiaGraph(
        precision=precision,
        order=order,
        inverse_order=inverse_order,
        parents=original_parents,
        coefficients=tuple(coefficients),
        conditional_variance=conditional_variance,
        metadata={
            "method": "vecchia",
            "nodes": count,
            "max_parents": None if max_parents is None else int(max_parents),
            "ordering": ordering if isinstance(ordering, str) else "custom",
            "workers": int(workers),
            "nonzeros": int(precision.nnz),
            "stabilised_nodes": int(stabilised_nodes),
        },
    )


def _selected_inverse_diagonal(factor, size, indices, chunk_size=64):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    result = np.empty(indices.size, dtype=float)
    for start in range(0, indices.size, chunk_size):
        selected = indices[start:start + chunk_size]
        right_hand_side = np.zeros((size, selected.size), dtype=float)
        right_hand_side[selected, np.arange(selected.size)] = 1.0
        solved = factor.solve(right_hand_side)
        result[start:start + selected.size] = solved[
            selected, np.arange(selected.size)]
    return result


def _hutchinson_inverse_diagonal(factor, size, samples, random_state):
    random = np.random.default_rng(random_state)
    estimate = np.zeros(size, dtype=float)
    for _ in range(samples):
        probe = random.choice(np.array([-1.0, 1.0]), size=size)
        estimate += probe * factor.solve(probe)
    return estimate / float(samples)


def _interval_tilted_moments(mean, variance, evidence):
    standard_deviation = np.sqrt(max(float(variance), np.finfo(float).eps))
    alpha = (evidence.lower - mean) / standard_deviation
    beta = (evidence.upper - mean) / standard_deviation
    if alpha > 8.0 or beta < -8.0:
        standardized_mean, standardized_variance = np.nan, np.nan
    else:
        standardized_mean, standardized_variance = truncnorm.stats(
            alpha, beta, moments="mv")
    invalid_moments = (
        not np.isfinite(standardized_mean)
        or not np.isfinite(standardized_variance)
        or standardized_variance <= np.finfo(float).eps
        or standardized_mean < alpha
        or standardized_mean > beta
    )
    if invalid_moments:
        reflected = beta < 0
        lower = -beta if reflected else alpha
        upper = -alpha if reflected else beta
        if lower < 0:
            raise FloatingPointError(
                "Interval evidence produced non-finite truncated-normal moments.")
        width = upper - lower
        scale = max(lower, 1.0)
        upper_scaled = 50.0 if not np.isfinite(width) else min(width * scale, 50.0)

        def kernel(scaled_offset):
            return np.exp(
                -lower * scaled_offset / scale
                -0.5 * (scaled_offset / scale) ** 2)

        zeroth = quad(
            kernel, 0.0, upper_scaled, epsabs=1e-13, epsrel=1e-11, limit=100)[0]
        first = quad(
            lambda offset: offset * kernel(offset),
            0.0, upper_scaled, epsabs=1e-13, epsrel=1e-11, limit=100)[0]
        second = quad(
            lambda offset: offset * offset * kernel(offset),
            0.0, upper_scaled, epsabs=1e-13, epsrel=1e-11, limit=100)[0]
        if not np.isfinite(zeroth) or zeroth <= 0:
            raise FloatingPointError(
                "Interval evidence produced non-finite truncated-normal moments "
                "for standardized bounds (%r, %r)." % (alpha, beta))
        offset_mean = first / zeroth / scale
        standardized_mean = lower + offset_mean
        standardized_variance = max(
            second / zeroth / (scale * scale) - offset_mean * offset_mean,
            np.finfo(float).eps,
        )
        if reflected:
            standardized_mean = -standardized_mean
    tilted_mean = mean + standard_deviation * float(standardized_mean)
    tilted_variance = variance * float(standardized_variance)
    tilted_variance = max(float(tilted_variance), np.finfo(float).eps)
    return float(tilted_mean), tilted_variance


def evidence_logpdf(evidence, values):
    """Evaluate a typed soft likelihood in log-density form."""
    values = np.asarray(values, dtype=float)
    if isinstance(evidence, GaussianEvidence):
        return norm.logpdf(values, loc=evidence.mean, scale=np.sqrt(evidence.variance))
    if isinstance(evidence, (IntervalEvidence, CensoredEvidence)):
        inside = (values >= evidence.lower) & (values <= evidence.upper)
        return np.where(inside, 0.0, -np.inf)
    if isinstance(evidence, SkewNormalEvidence):
        return skewnorm.logpdf(
            values, evidence.shape, loc=evidence.location, scale=evidence.scale)
    if isinstance(evidence, LogNormalEvidence):
        return lognorm.logpdf(
            values, s=evidence.log_scale, scale=np.exp(evidence.log_mean))
    if isinstance(evidence, StudentTEvidence):
        return student_t.logpdf(
            values, df=evidence.degrees_of_freedom,
            loc=evidence.location, scale=evidence.scale)
    if isinstance(evidence, GaussianMixtureEvidence):
        component_logs = np.stack([
            np.log(weight) + norm.logpdf(values, loc=mean, scale=np.sqrt(variance))
            for weight, mean, variance in zip(
                evidence.weights, evidence.means, evidence.variances)
            if weight > 0
        ], axis=0)
        return logsumexp(component_logs, axis=0)
    if isinstance(evidence, TabulatedEvidence):
        abscissa = np.asarray(evidence.abscissa, dtype=float)
        density = np.asarray(evidence.density, dtype=float)
        interpolated = np.interp(values, abscissa, density, left=0.0, right=0.0)
        with np.errstate(divide="ignore"):
            return np.log(interpolated)
    raise TypeError("Unsupported soft evidence type.")


def evidence_pdf(evidence, values):
    """Evaluate a typed soft likelihood as a density or indicator."""
    return np.exp(evidence_logpdf(evidence, values))


def _gaussian_mixture_tilted_moments(mean, variance, evidence):
    weights = np.asarray(evidence.weights, dtype=float)
    component_mean = np.asarray(evidence.means, dtype=float)
    component_variance = np.asarray(evidence.variances, dtype=float)
    tilted_variance = 1.0 / (1.0 / variance + 1.0 / component_variance)
    tilted_mean = tilted_variance * (
        mean / variance + component_mean / component_variance)
    log_weight = (
        np.log(weights)
        + norm.logpdf(mean, loc=component_mean, scale=np.sqrt(variance + component_variance))
    )
    normalized_weight = np.exp(log_weight - logsumexp(log_weight))
    mixture_mean = float(np.dot(normalized_weight, tilted_mean))
    mixture_second = float(np.dot(
        normalized_weight, tilted_variance + tilted_mean * tilted_mean))
    mixture_variance = max(
        mixture_second - mixture_mean * mixture_mean, np.finfo(float).eps)
    effective_components = float(1.0 / np.sum(normalized_weight * normalized_weight))
    return mixture_mean, mixture_variance, {
        "moment_engine": "analytic_gaussian_mixture",
        "relative_quadrature_error": 0.0,
        "effective_components": effective_components,
    }


def _quadrature_tilted_moments(mean, variance, evidence):
    standard_deviation = np.sqrt(max(float(variance), np.finfo(float).eps))
    if isinstance(evidence, TabulatedEvidence):
        x = np.asarray(evidence.abscissa, dtype=float)
        likelihood = np.asarray(evidence.density, dtype=float)
        refined_x = np.concatenate([
            np.linspace(left, right, 9, endpoint=False)
            for left, right in zip(x[:-1], x[1:])
        ] + [x[-1:]])
        refined_likelihood = np.interp(refined_x, x, likelihood)
        kernel = (
            norm.pdf(refined_x, loc=mean, scale=standard_deviation)
            * refined_likelihood)
        zeroth = float(simpson(kernel, x=refined_x))
        if not np.isfinite(zeroth) or zeroth <= np.finfo(float).tiny:
            raise FloatingPointError("Tabulated likelihood has negligible overlap with the EP cavity.")
        tilted_mean = float(simpson(refined_x * kernel, x=refined_x) / zeroth)
        tilted_variance = max(
            float(simpson(
                (refined_x - tilted_mean) ** 2 * kernel, x=refined_x) / zeroth),
            np.finfo(float).eps,
        )
        coarse_x = x
        coarse_kernel = norm.pdf(
            coarse_x, loc=mean, scale=standard_deviation) * likelihood
        coarse_zeroth = float(simpson(coarse_kernel, x=coarse_x))
        coarse_mean = float(simpson(coarse_x * coarse_kernel, x=coarse_x) / coarse_zeroth)
        coarse_variance = float(simpson(
            (coarse_x - coarse_mean) ** 2 * coarse_kernel, x=coarse_x) / coarse_zeroth)
        relative_error = max(
            abs(coarse_mean - tilted_mean) / max(abs(tilted_mean), standard_deviation),
            abs(coarse_variance - tilted_variance) / tilted_variance,
        )
        return tilted_mean, tilted_variance, {
            "moment_engine": "tabulated_trapezoid",
            "relative_quadrature_error": float(relative_error),
            "effective_components": 1.0,
        }
    def gauss_legendre(nodes, weights):
        standardized = 12.0 * nodes
        values = mean + standard_deviation * standardized
        log_likelihood = evidence_logpdf(evidence, values)
        finite = np.isfinite(log_likelihood)
        if not np.any(finite):
            raise FloatingPointError("Soft likelihood has negligible overlap with the EP cavity.")
        log_scale = float(np.max(log_likelihood[finite]))
        tilted_weight = np.zeros_like(weights)
        tilted_weight[finite] = (
            weights[finite] * norm.pdf(standardized[finite])
            * np.exp(log_likelihood[finite] - log_scale))
        normalization = float(np.sum(tilted_weight))
        if normalization <= np.finfo(float).tiny:
            raise FloatingPointError("Tilted-moment normalization is non-positive.")
        tilted_weight /= normalization
        center = float(np.dot(tilted_weight, values))
        spread = max(
            float(np.dot(tilted_weight, (values - center) ** 2)),
            np.finfo(float).eps,
        )
        return center, spread

    coarse_mean, coarse_variance = gauss_legendre(
        _GL_NODES_256, _GL_WEIGHTS_256)
    tilted_mean, tilted_variance = gauss_legendre(
        _GL_NODES_512, _GL_WEIGHTS_512)
    relative_error = max(
        abs(coarse_mean - tilted_mean) / max(abs(tilted_mean), standard_deviation),
        abs(coarse_variance - tilted_variance) / tilted_variance,
    )
    return tilted_mean, tilted_variance, {
        "moment_engine": "gauss_legendre_512",
        "relative_quadrature_error": float(relative_error),
        "effective_components": 1.0,
    }

def _tilted_moments(mean, variance, evidence):
    if isinstance(evidence, (IntervalEvidence, CensoredEvidence)):
        tilted_mean, tilted_variance = _interval_tilted_moments(
            mean, variance, evidence)
        return tilted_mean, tilted_variance, {
            "moment_engine": "analytic_truncated_normal",
            "relative_quadrature_error": 0.0,
            "effective_components": 1.0,
        }
    if isinstance(evidence, GaussianMixtureEvidence):
        return _gaussian_mixture_tilted_moments(mean, variance, evidence)
    return _quadrature_tilted_moments(mean, variance, evidence)


def _evidence_variance_scale(evidence, fallback):
    if isinstance(evidence, IntervalEvidence):
        return (evidence.upper - evidence.lower) ** 2 / 12.0
    if isinstance(evidence, CensoredEvidence):
        return fallback
    if isinstance(evidence, GaussianMixtureEvidence):
        weights = np.asarray(evidence.weights, dtype=float)
        means = np.asarray(evidence.means, dtype=float)
        variances = np.asarray(evidence.variances, dtype=float)
        center = float(np.dot(weights, means))
        return float(np.dot(weights, variances + (means - center) ** 2))
    if isinstance(evidence, SkewNormalEvidence):
        delta = evidence.shape / np.sqrt(1.0 + evidence.shape ** 2)
        return evidence.scale ** 2 * (1.0 - 2.0 * delta ** 2 / np.pi)
    if isinstance(evidence, LogNormalEvidence):
        sigma2 = evidence.log_scale ** 2
        return (np.exp(sigma2) - 1.0) * np.exp(2.0 * evidence.log_mean + sigma2)
    if isinstance(evidence, StudentTEvidence):
        return evidence.scale ** 2 * evidence.degrees_of_freedom / (
            evidence.degrees_of_freedom - 2.0)
    if isinstance(evidence, TabulatedEvidence):
        x = np.asarray(evidence.abscissa, dtype=float)
        density = np.asarray(evidence.density, dtype=float)
        center = float(trapezoid(x * density, x))
        return float(trapezoid((x - center) ** 2 * density, x))
    return fallback


class VISTAOperator:
    """Compiled exact or Vecchia VISTA-BME operator."""

    def __init__(
            self, ck, ch, cs, soft_evidence, covmodel, covparam,
            hard_variance=0.0, trend="zero", mode="auto",
            max_parents=32, ordering="random", random_state=0,
            workers=1, ep_damping=0.5, ep_tolerance=1e-6,
            ep_max_iterations=50, variance_exact_limit=3000,
            variance_samples=64, ep_reuse_converged_factor=True):
        self.ck = _as_coordinates(ck, "ck")
        dimensions = self.ck.shape[1]
        self.ch = _as_coordinates(
            ch, "ch", dimensions=dimensions, allow_empty=True)
        self.cs = _as_coordinates(
            cs, "cs", dimensions=dimensions, allow_empty=True)
        if self.ch.shape[0] + self.cs.shape[0] == 0:
            raise ValueError("At least one hard or soft observation is required.")
        self.soft_evidence = tuple(() if soft_evidence is None else soft_evidence)
        if len(self.soft_evidence) != self.cs.shape[0]:
            raise ValueError("soft_evidence must match the number of soft coordinates.")
        if not all(isinstance(item, SOFT_EVIDENCE_TYPES) for item in self.soft_evidence):
            raise TypeError("soft_evidence contains an unsupported evidence type.")
        self.hard_variance = _broadcast_variance(
            hard_variance, self.ch.shape[0], "hard_variance")
        self.covmodel = covmodel
        self.covparam = covparam
        self.trend = trend
        self.max_parents = max_parents
        self.ordering = ordering
        self.random_state = int(random_state)
        self.workers = int(workers)
        self.ep_damping = float(ep_damping)
        self.ep_tolerance = float(ep_tolerance)
        self.ep_max_iterations = int(ep_max_iterations)
        self.variance_exact_limit = int(variance_exact_limit)
        self.variance_samples = int(variance_samples)
        self.ep_reuse_converged_factor = bool(ep_reuse_converged_factor)
        if not 0 < self.ep_damping <= 1:
            raise ValueError("ep_damping must be in (0, 1].")
        if self.ep_tolerance <= 0 or self.ep_max_iterations < 1:
            raise ValueError("EP tolerance and iteration count must be positive.")
        if self.variance_exact_limit < 1 or self.variance_samples < 1:
            raise ValueError("Variance controls must be positive.")

        all_gaussian = all(
            isinstance(item, GaussianEvidence) for item in self.soft_evidence)
        normalised_mode = str(mode).strip().lower()
        if normalised_mode == "auto":
            normalised_mode = "exact" if max_parents is None and all_gaussian else "vecchia"
        if normalised_mode not in ("exact", "vecchia"):
            raise ValueError("mode must be 'auto', 'exact', or 'vecchia'.")
        if normalised_mode == "exact" and not all_gaussian:
            raise ValueError("Exact mode currently requires Gaussian soft evidence.")
        if normalised_mode == "vecchia" and not _is_zero_trend(trend):
            raise ValueError("Vecchia inference currently requires a zero prior-mean trend.")
        self.mode = normalised_mode

        self._exact_operator = None
        self.graph = None
        if self.mode == "exact":
            soft_variance = np.array(
                [item.variance for item in self.soft_evidence], dtype=float)
            self._exact_operator = compile_bme_operator(
                self.ck,
                ch=self.ch,
                cs=self.cs,
                covmodel=self.covmodel,
                covparam=self.covparam,
                nhmax=self.ch.shape[0],
                nsmax=self.cs.shape[0],
                dmax=None,
                soft_variance=soft_variance,
                hard_variance=self.hard_variance,
                trend=self.trend,
                backend="cpu",
            )
        else:
            self._all_coordinates = np.vstack((self.ch, self.cs, self.ck))
            self.graph = build_vecchia_graph(
                self._all_coordinates,
                self.covmodel,
                self.covparam,
                max_parents=self.max_parents,
                ordering=self.ordering,
                random_state=self.random_state,
                workers=self.workers,
            )

    @property
    def n_targets(self):
        return self.ck.shape[0]

    def predict(self, hard_values=None, soft_evidence=None):
        hard_values = _as_values(
            hard_values, self.ch.shape[0], "hard_values")
        evidence = tuple(self.soft_evidence if soft_evidence is None else soft_evidence)
        if len(evidence) != self.cs.shape[0]:
            raise ValueError("soft_evidence must match the number of soft coordinates.")
        if self.mode == "exact":
            if not all(isinstance(item, GaussianEvidence) for item in evidence):
                raise ValueError("Exact mode requires Gaussian soft evidence.")
            expected_variance = np.array(
                [item.variance for item in self.soft_evidence], dtype=float)
            actual_variance = np.array(
                [item.variance for item in evidence], dtype=float)
            if not np.array_equal(expected_variance, actual_variance):
                raise ValueError("Changing Gaussian soft variances requires recompilation.")
            soft_mean = np.array([item.mean for item in evidence], dtype=float)
            mean = np.asarray(
                self._exact_operator.predict(hard_values, soft_mean),
                dtype=float).reshape(-1)
            return VISTAResult(
                mean=mean,
                variance=np.asarray(
                    self._exact_operator.variance, dtype=float).reshape(-1),
                metadata={
                    "name": "VISTA-BME",
                    "mode": "exact_gaussian",
                    "exact_limit": True,
                    "backend": self._exact_operator.metadata.get("backend", "cpu"),
                    "ep_iterations": 0,
                    "ep_converged": True,
                },
            )
        return self._predict_vecchia(hard_values, evidence)

    def predict_mixture(
            self, hard_values=None, soft_evidence=None, max_components=128):
        """Preserve Gaussian-mixture soft evidence in target marginals.

        The calculation enumerates a bounded set of likelihood-component
        combinations and performs one sparse Gaussian update per retained
        combination.  It is intended for multimodal soft evidence with a small
        component budget; ordinary ``predict`` remains the scalable EP path.
        """
        if self.mode != "vecchia":
            raise ValueError("Mixture-preserving prediction requires Vecchia mode.")
        hard_values = _as_values(
            hard_values, self.ch.shape[0], "hard_values")
        evidence = tuple(self.soft_evidence if soft_evidence is None else soft_evidence)
        if len(evidence) != self.cs.shape[0]:
            raise ValueError("soft_evidence must match the number of soft coordinates.")
        if not all(isinstance(item, (GaussianEvidence, GaussianMixtureEvidence))
                   for item in evidence):
            raise TypeError(
                "Mixture-preserving prediction accepts Gaussian and GaussianMixtureEvidence.")
        max_components = int(max_components)
        if max_components < 1:
            raise ValueError("max_components must be positive.")

        hard_count = self.ch.shape[0]
        soft_count = self.cs.shape[0]
        total_count = self.graph.precision.shape[0]
        hard_indices = np.arange(hard_count, dtype=np.int64)
        soft_indices = np.arange(
            hard_count, hard_count + soft_count, dtype=np.int64)
        target_indices = np.arange(
            hard_count + soft_count, total_count, dtype=np.int64)

        fixed_mask = self.hard_variance == 0
        fixed_indices = hard_indices[fixed_mask]
        free_mask = np.ones(total_count, dtype=bool)
        free_mask[fixed_indices] = False
        free_indices = np.flatnonzero(free_mask)
        global_to_free = np.full(total_count, -1, dtype=np.int64)
        global_to_free[free_indices] = np.arange(free_indices.size, dtype=np.int64)
        target_local = global_to_free[target_indices]

        precision = self.graph.precision
        base_precision = precision[free_indices, :][:, free_indices].tocsr()
        base_rhs = np.zeros(free_indices.size, dtype=float)
        if fixed_indices.size:
            base_rhs -= (
                precision[free_indices, :][:, fixed_indices]
                @ hard_values[fixed_mask]
            )
        base_tau = np.zeros(free_indices.size, dtype=float)
        noisy_hard_indices = hard_indices[~fixed_mask]
        if noisy_hard_indices.size:
            local = global_to_free[noisy_hard_indices]
            hard_noise = self.hard_variance[~fixed_mask]
            base_tau[local] += 1.0 / hard_noise
            base_rhs[local] += hard_values[~fixed_mask] / hard_noise
        base_precision = base_precision + diags(base_tau)

        component_sets = []
        for item in evidence:
            if isinstance(item, GaussianEvidence):
                component_sets.append(((1.0, item.mean, item.variance),))
            else:
                component_sets.append(tuple(zip(
                    item.weights, item.means, item.variances)))
        combinations = []
        for combination in product(*component_sets):
            prior_weight = float(np.prod([part[0] for part in combination]))
            combinations.append((prior_weight, combination))
        combinations.sort(key=lambda entry: entry[0], reverse=True)
        retained = combinations[:max_components]
        retained_prior_mass = float(sum(entry[0] for entry in retained))

        log_weights = []
        component_target_mean = []
        component_target_variance = []
        soft_local = global_to_free[soft_indices]
        for prior_weight, combination in retained:
            tau = np.zeros(free_indices.size, dtype=float)
            rhs = base_rhs.copy()
            constant = 0.0
            for local_index, (_, mean, variance) in zip(soft_local, combination):
                tau[local_index] += 1.0 / variance
                rhs[local_index] += mean / variance
                constant += np.log(2.0 * np.pi * variance) + mean * mean / variance
            posterior_precision = base_precision + diags(tau)
            factor = splu(posterior_precision.tocsc())
            posterior_mean = factor.solve(rhs)
            diagonal = _selected_inverse_diagonal(
                factor, free_indices.size, target_local)
            log_determinant = float(np.sum(np.log(np.abs(factor.U.diagonal()))))
            log_weights.append(
                np.log(prior_weight)
                - 0.5 * constant
                - 0.5 * log_determinant
                + 0.5 * float(np.dot(rhs, posterior_mean))
            )
            component_target_mean.append(posterior_mean[target_local])
            component_target_variance.append(diagonal)

        normalized_weights = np.exp(
            np.asarray(log_weights) - logsumexp(log_weights))
        component_target_mean = np.asarray(component_target_mean, dtype=float)
        component_target_variance = np.asarray(
            component_target_variance, dtype=float)
        posterior_mean = normalized_weights @ component_target_mean
        posterior_second = normalized_weights @ (
            component_target_variance + component_target_mean ** 2)
        posterior_variance = np.maximum(
            posterior_second - posterior_mean ** 2, np.finfo(float).eps)
        effective_components = float(
            1.0 / np.sum(normalized_weights * normalized_weights))
        return VISTAMixtureResult(
            mean=posterior_mean,
            variance=posterior_variance,
            weights=normalized_weights,
            component_mean=component_target_mean,
            component_variance=component_target_variance,
            metadata={
                "name": "VISTA-BME",
                "mode": "vecchia_mixture",
                "exact_limit": self.max_parents is None,
                "max_parents": self.max_parents,
                "graph_nonzeros": int(self.graph.precision.nnz),
                "retained_components": int(len(retained)),
                "candidate_components": int(len(combinations)),
                "retained_prior_mass": retained_prior_mass,
                "discarded_prior_mass": max(0.0, 1.0 - retained_prior_mass),
                "effective_posterior_components": effective_components,
                "variance_method": "selected_inverse_exact",
            },
        )

    def _predict_vecchia(self, hard_values, evidence):
        if not all(isinstance(item, SOFT_EVIDENCE_TYPES) for item in evidence):
            raise TypeError("Unsupported soft evidence type.")
        hard_count = self.ch.shape[0]
        soft_count = self.cs.shape[0]
        total_count = self.graph.precision.shape[0]
        hard_indices = np.arange(hard_count, dtype=np.int64)
        soft_indices = np.arange(
            hard_count, hard_count + soft_count, dtype=np.int64)
        target_indices = np.arange(
            hard_count + soft_count, total_count, dtype=np.int64)

        fixed_mask = self.hard_variance == 0
        fixed_indices = hard_indices[fixed_mask]
        free_mask = np.ones(total_count, dtype=bool)
        free_mask[fixed_indices] = False
        free_indices = np.flatnonzero(free_mask)
        global_to_free = np.full(total_count, -1, dtype=np.int64)
        global_to_free[free_indices] = np.arange(free_indices.size, dtype=np.int64)

        precision = self.graph.precision
        free_precision = precision[free_indices, :][:, free_indices].tocsr()
        right_hand_side = np.zeros(free_indices.size, dtype=float)
        if fixed_indices.size:
            fixed_values = hard_values[fixed_mask]
            right_hand_side -= precision[free_indices, :][:, fixed_indices] @ fixed_values

        base_tau = np.zeros(free_indices.size, dtype=float)
        base_nu = np.zeros(free_indices.size, dtype=float)
        noisy_hard_indices = hard_indices[~fixed_mask]
        if noisy_hard_indices.size:
            local = global_to_free[noisy_hard_indices]
            variance = self.hard_variance[~fixed_mask]
            base_tau[local] += 1.0 / variance
            base_nu[local] += hard_values[~fixed_mask] / variance

        non_gaussian_global = []
        non_gaussian_evidence = []
        for global_index, item in zip(soft_indices, evidence):
            local_index = global_to_free[global_index]
            if isinstance(item, GaussianEvidence):
                base_tau[local_index] += 1.0 / item.variance
                base_nu[local_index] += item.mean / item.variance
            else:
                non_gaussian_global.append(int(global_index))
                non_gaussian_evidence.append(item)

        non_gaussian_local = global_to_free[
            np.asarray(non_gaussian_global, dtype=np.int64)
        ] if non_gaussian_global else np.empty(0, dtype=np.int64)
        site_tau = np.zeros(non_gaussian_local.size, dtype=float)
        site_nu = np.zeros(non_gaussian_local.size, dtype=float)
        converged = non_gaussian_local.size == 0
        maximum_change = 0.0
        iterations = 0
        clipped_sites = 0
        maximum_quadrature_error = 0.0
        moment_engines = set()
        maximum_effective_components = 1.0

        factor = None
        posterior_mean = None
        factorizations = 0
        reused_converged_factor = False
        for iteration in range(1, self.ep_max_iterations + 1):
            tau = base_tau.copy()
            nu = base_nu.copy()
            if non_gaussian_local.size:
                tau[non_gaussian_local] += site_tau
                nu[non_gaussian_local] += site_nu
            posterior_precision = free_precision + diags(tau)
            factor = splu(posterior_precision.tocsc())
            factorizations += 1
            posterior_mean = factor.solve(right_hand_side + nu)
            iterations = iteration if non_gaussian_local.size else 0
            if not non_gaussian_local.size:
                reused_converged_factor = self.ep_reuse_converged_factor
                break
            marginal_variance = _selected_inverse_diagonal(
                factor, free_indices.size, non_gaussian_local)
            new_tau = site_tau.copy()
            new_nu = site_nu.copy()
            for site_index, (local_index, item) in enumerate(
                    zip(non_gaussian_local, non_gaussian_evidence)):
                variance = max(float(marginal_variance[site_index]), np.finfo(float).eps)
                marginal_precision = 1.0 / variance
                cavity_precision = marginal_precision - site_tau[site_index]
                if not np.isfinite(cavity_precision) or cavity_precision <= 1e-12:
                    raise FloatingPointError("EP produced non-positive cavity precision.")
                cavity_nu = posterior_mean[local_index] / variance - site_nu[site_index]
                cavity_variance = 1.0 / cavity_precision
                cavity_mean = cavity_nu / cavity_precision
                tilted_mean, tilted_variance, moment_diagnostics = _tilted_moments(
                    cavity_mean, cavity_variance, item)
                moment_engines.add(moment_diagnostics["moment_engine"])
                maximum_quadrature_error = max(
                    maximum_quadrature_error,
                    moment_diagnostics["relative_quadrature_error"],
                )
                maximum_effective_components = max(
                    maximum_effective_components,
                    moment_diagnostics["effective_components"],
                )
                proposed_tau = max(1.0 / tilted_variance - cavity_precision, 0.0)
                proposed_nu = tilted_mean / tilted_variance - cavity_nu
                variance_scale = max(
                    _evidence_variance_scale(item, cavity_variance),
                    np.finfo(float).eps,
                )
                precision_cap = 1e8 / variance_scale
                if proposed_tau > precision_cap:
                    proposed_tau = precision_cap
                    site_center = tilted_mean
                    if isinstance(item, (IntervalEvidence, CensoredEvidence)):
                        site_center = np.clip(tilted_mean, item.lower, item.upper)
                    proposed_nu = proposed_tau * float(site_center)
                    clipped_sites += 1
                new_tau[site_index] = (
                    (1.0 - self.ep_damping) * site_tau[site_index]
                    + self.ep_damping * proposed_tau)
                new_nu[site_index] = (
                    (1.0 - self.ep_damping) * site_nu[site_index]
                    + self.ep_damping * proposed_nu)
            maximum_change = float(max(
                np.max(np.abs(new_tau - site_tau)),
                np.max(np.abs(new_nu - site_nu)),
            ))
            if maximum_change <= self.ep_tolerance:
                converged = True
                if self.ep_reuse_converged_factor:
                    # The current factor already represents site parameters whose
                    # proposed update is below tolerance.  Reusing it avoids a
                    # numerically redundant sparse factorization.
                    reused_converged_factor = True
                else:
                    site_tau, site_nu = new_tau, new_nu
                break
            site_tau, site_nu = new_tau, new_nu

        if not reused_converged_factor:
            # Re-factor when the iteration cap is reached or when legacy exact
            # final-site semantics are explicitly requested.
            tau = base_tau.copy()
            nu = base_nu.copy()
            if non_gaussian_local.size:
                tau[non_gaussian_local] += site_tau
                nu[non_gaussian_local] += site_nu
            posterior_precision = free_precision + diags(tau)
            factor = splu(posterior_precision.tocsc())
            factorizations += 1
            posterior_mean = factor.solve(right_hand_side + nu)

        ep_mean_residual = 0.0
        ep_variance_residual = 0.0
        if non_gaussian_local.size:
            marginal_variance = _selected_inverse_diagonal(
                factor, free_indices.size, non_gaussian_local)
            for site_index, (local_index, item) in enumerate(
                    zip(non_gaussian_local, non_gaussian_evidence)):
                variance = max(
                    float(marginal_variance[site_index]), np.finfo(float).eps)
                marginal_precision = 1.0 / variance
                cavity_precision = marginal_precision - site_tau[site_index]
                if cavity_precision <= 0:
                    ep_mean_residual = np.inf
                    ep_variance_residual = np.inf
                    break
                cavity_nu = (
                    posterior_mean[local_index] / variance - site_nu[site_index])
                cavity_variance = 1.0 / cavity_precision
                cavity_mean = cavity_nu / cavity_precision
                tilted_mean, tilted_variance, moment_diagnostics = _tilted_moments(
                    cavity_mean, cavity_variance, item)
                moment_engines.add(moment_diagnostics["moment_engine"])
                maximum_quadrature_error = max(
                    maximum_quadrature_error,
                    moment_diagnostics["relative_quadrature_error"],
                )
                maximum_effective_components = max(
                    maximum_effective_components,
                    moment_diagnostics["effective_components"],
                )
                ep_mean_residual = max(
                    ep_mean_residual,
                    abs(float(posterior_mean[local_index]) - tilted_mean)
                    / np.sqrt(tilted_variance),
                )
                ep_variance_residual = max(
                    ep_variance_residual,
                    abs(variance - tilted_variance) / tilted_variance,
                )

        target_local = global_to_free[target_indices]
        if free_indices.size <= self.variance_exact_limit:
            target_variance = _selected_inverse_diagonal(
                factor, free_indices.size, target_local)
            variance_method = "selected_inverse_exact"
        else:
            diagonal_estimate = _hutchinson_inverse_diagonal(
                factor, free_indices.size, self.variance_samples,
                self.random_state)
            target_variance = np.maximum(
                diagonal_estimate[target_local], np.finfo(float).eps)
            variance_method = "hutchinson_%d" % self.variance_samples

        return VISTAResult(
            mean=np.asarray(posterior_mean[target_local], dtype=float),
            variance=np.asarray(target_variance, dtype=float),
            metadata={
                "name": "VISTA-BME",
                "mode": "vecchia_ep" if non_gaussian_local.size else "vecchia_gaussian",
                "exact_limit": self.max_parents is None,
                "max_parents": self.max_parents,
                "graph_nonzeros": int(self.graph.precision.nnz),
                "ep_iterations": int(iterations),
                "ep_converged": bool(converged),
                "ep_max_site_change": float(maximum_change),
                "ep_standardized_mean_residual": float(ep_mean_residual),
                "ep_relative_variance_residual": float(ep_variance_residual),
                "ep_precision_clips": int(clipped_sites),
                "factorizations": int(factorizations),
                "ep_reused_converged_factor": bool(reused_converged_factor),
                "non_gaussian_sites": int(non_gaussian_local.size),
                "interval_sites": int(sum(isinstance(
                    item, (IntervalEvidence, CensoredEvidence))
                    for item in non_gaussian_evidence)),
                "soft_evidence_families": {
                    evidence_type.__name__: int(sum(
                        isinstance(item, evidence_type) for item in evidence))
                    for evidence_type in SOFT_EVIDENCE_TYPES
                    if any(isinstance(item, evidence_type) for item in evidence)
                },
                "moment_engines": sorted(moment_engines),
                "maximum_relative_quadrature_error": float(maximum_quadrature_error),
                "maximum_effective_components": float(maximum_effective_components),
                "variance_method": variance_method,
            },
        )


def compile_vista_operator(
        ck, ch=None, cs=None, soft_evidence=None,
        covmodel=None, covparam=None, **kwargs):
    """Compile a VISTA-BME exact or sparse inference operator."""
    if covmodel is None or covparam is None:
        raise ValueError("covmodel and covparam are required.")
    return VISTAOperator(
        ck=ck,
        ch=ch,
        cs=cs,
        soft_evidence=soft_evidence,
        covmodel=covmodel,
        covparam=covparam,
        **kwargs
    )


@dataclass
class StreamingRankState:
    """Dense reference implementation of exact rank-one observation updates.

    This class is the correctness oracle for the future sparse Cholesky update.
    It conditions a fixed latent support on one new noisy observation without
    rebuilding the batch posterior.
    """

    mean: np.ndarray
    covariance: np.ndarray
    updates: int = 0

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=float).reshape(-1).copy()
        self.covariance = np.asarray(self.covariance, dtype=float).copy()
        if self.covariance.shape != (self.mean.size, self.mean.size):
            raise ValueError("covariance must be square and match the mean.")
        if not np.allclose(self.covariance, self.covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("covariance must be symmetric.")

    def update(self, node_index, value, variance=0.0):
        """Assimilate one observation and return innovation diagnostics."""
        node_index = int(node_index)
        if node_index < 0 or node_index >= self.mean.size:
            raise IndexError("node_index is outside the latent support.")
        if not np.isfinite(value):
            raise ValueError("value must be finite.")
        if not np.isfinite(variance) or variance < 0:
            raise ValueError("variance must be finite and non-negative.")
        cross_covariance = self.covariance[:, node_index].copy()
        innovation_variance = float(
            self.covariance[node_index, node_index] + variance)
        if innovation_variance <= np.finfo(float).eps:
            raise np.linalg.LinAlgError("Observation has no remaining innovation variance.")
        innovation = float(value - self.mean[node_index])
        gain = cross_covariance / innovation_variance
        self.mean += gain * innovation
        self.covariance -= np.outer(
            cross_covariance, cross_covariance) / innovation_variance
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.updates += 1
        return {
            "innovation": innovation,
            "innovation_variance": innovation_variance,
            "standardized_innovation": innovation / np.sqrt(innovation_variance),
            "updates": self.updates,
        }


class StreamingPrecisionState:
    """Exact low-rank streaming updates over a fixed sparse precision graph.

    The sparse base precision is factorized once. Each new observation needs
    one sparse solve and a Cholesky border update of the small observation
    system; the base Vecchia graph is not rebuilt. The retained low-rank basis
    grows with the number of updates and should be periodically rebased in a
    long-running service.
    """

    def __init__(self, precision, prior_mean=None):
        self.precision = csr_matrix(precision, dtype=float)
        if self.precision.shape[0] != self.precision.shape[1]:
            raise ValueError("precision must be square.")
        self.size = int(self.precision.shape[0])
        self.prior_mean = (
            np.zeros(self.size, dtype=float)
            if prior_mean is None
            else _as_values(prior_mean, self.size, "prior_mean").copy()
        )
        self.mean = self.prior_mean.copy()
        self._factor = splu(self.precision.tocsc())
        self._base_cross = np.empty((self.size, 0), dtype=float)
        self._observation_cholesky = np.empty((0, 0), dtype=float)
        self.node_indices = []
        self.noise_variances = []

    @property
    def updates(self):
        return len(self.node_indices)

    def _base_covariance_column(self, node_index):
        right_hand_side = np.zeros(self.size, dtype=float)
        right_hand_side[node_index] = 1.0
        return np.asarray(self._factor.solve(right_hand_side), dtype=float)

    def update(self, node_index, value, variance=0.0):
        """Assimilate a scalar observation without refactorizing the graph."""
        node_index = int(node_index)
        if node_index < 0 or node_index >= self.size:
            raise IndexError("node_index is outside the latent support.")
        if not np.isfinite(value):
            raise ValueError("value must be finite.")
        if not np.isfinite(variance) or variance < 0:
            raise ValueError("variance must be finite and non-negative.")

        base_column = self._base_covariance_column(node_index)
        if self.updates:
            base_row = self._base_cross[node_index, :]
            whitened = solve_triangular(
                self._observation_cholesky,
                base_row,
                lower=True,
                check_finite=False,
            )
            system_weights = solve_triangular(
                self._observation_cholesky.T,
                whitened,
                lower=False,
                check_finite=False,
            )
            posterior_cross = base_column - self._base_cross @ system_weights
        else:
            base_row = np.empty(0, dtype=float)
            whitened = np.empty(0, dtype=float)
            posterior_cross = base_column

        innovation_variance = float(posterior_cross[node_index] + variance)
        tolerance = np.finfo(float).eps * max(1.0, abs(base_column[node_index]))
        if innovation_variance <= tolerance:
            raise np.linalg.LinAlgError(
                "Observation has no remaining innovation variance.")
        innovation = float(value - self.mean[node_index])
        self.mean += posterior_cross * (innovation / innovation_variance)

        old_size = self.updates
        expanded = np.zeros((old_size + 1, old_size + 1), dtype=float)
        if old_size:
            expanded[:old_size, :old_size] = self._observation_cholesky
            expanded[old_size, :old_size] = whitened
        expanded[old_size, old_size] = np.sqrt(innovation_variance)
        self._observation_cholesky = expanded
        self._base_cross = np.column_stack((self._base_cross, base_column))
        self.node_indices.append(node_index)
        self.noise_variances.append(float(variance))
        return {
            "innovation": innovation,
            "innovation_variance": innovation_variance,
            "standardized_innovation": innovation / np.sqrt(innovation_variance),
            "updates": self.updates,
            "graph_refactorized": False,
            "rank": self.updates,
        }

    def posterior_variance(self, indices=None, chunk_size=64):
        """Return exact posterior marginal variances at selected graph nodes."""
        if indices is None:
            indices = np.arange(self.size, dtype=np.int64)
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise IndexError("indices contain a node outside the latent support.")
        base_variance = _selected_inverse_diagonal(
            self._factor, self.size, indices, chunk_size=chunk_size)
        if not self.updates:
            return base_variance
        whitened = solve_triangular(
            self._observation_cholesky,
            self._base_cross[indices, :].T,
            lower=True,
            check_finite=False,
        )
        posterior = base_variance - np.sum(whitened ** 2, axis=0)
        tolerance = 1e-12 * np.maximum(1.0, np.abs(base_variance))
        posterior[(posterior < 0) & (posterior >= -tolerance)] = 0.0
        if np.any(posterior < 0):
            raise FloatingPointError("Streaming update produced negative variance.")
        return posterior


class StreamingSiteAccumulator:
    """Constant-memory sufficient statistics for repeated Gaussian observations.

    Observations that share a fixed latent node contribute additively to the
    diagonal likelihood precision and natural mean. The retained state has one
    precision and one natural parameter per node, independent of stream length.
    """

    def __init__(self, size):
        self.size = int(size)
        if self.size < 1:
            raise ValueError("size must be positive.")
        self.precision = np.zeros(self.size, dtype=float)
        self.natural_mean = np.zeros(self.size, dtype=float)
        self.count = np.zeros(self.size, dtype=np.int64)
        self.updates = 0

    def update(self, node_index, value, variance):
        self.update_many([node_index], [value], [variance])

    def update_many(self, node_indices, values, variances):
        node_indices = np.asarray(node_indices, dtype=np.int64).reshape(-1)
        values = np.asarray(values, dtype=float).reshape(-1)
        variances = np.asarray(variances, dtype=float).reshape(-1)
        if not (node_indices.size == values.size == variances.size):
            raise ValueError("node_indices, values, and variances must have matching sizes.")
        if np.any(node_indices < 0) or np.any(node_indices >= self.size):
            raise IndexError("node_indices contain a node outside the latent support.")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must be finite.")
        if not np.all(np.isfinite(variances)) or np.any(variances <= 0):
            raise ValueError("variances must be finite and positive.")
        likelihood_precision = 1.0 / variances
        self.precision += np.bincount(
            node_indices, weights=likelihood_precision, minlength=self.size)
        self.natural_mean += np.bincount(
            node_indices, weights=likelihood_precision * values, minlength=self.size)
        self.count += np.bincount(node_indices, minlength=self.size).astype(np.int64)
        self.updates += int(node_indices.size)

    def posterior(self, prior_precision, prior_mean=None):
        prior_precision = csr_matrix(prior_precision, dtype=float)
        if prior_precision.shape != (self.size, self.size):
            raise ValueError("prior_precision must match the accumulator size.")
        prior_mean = (
            np.zeros(self.size, dtype=float)
            if prior_mean is None
            else _as_values(prior_mean, self.size, "prior_mean")
        )
        right_hand_side = prior_precision @ prior_mean + self.natural_mean
        posterior_precision = prior_precision + diags(self.precision)
        factor = splu(posterior_precision.tocsc())
        return np.asarray(factor.solve(right_hand_side), dtype=float), factor


def gaussian_exactness_report(reference, candidate):
    """Return transparent posterior-moment errors for a Gaussian gate."""
    reference_mean = np.asarray(reference.mean, dtype=float).reshape(-1)
    candidate_mean = np.asarray(candidate.mean, dtype=float).reshape(-1)
    reference_variance = np.asarray(reference.variance, dtype=float).reshape(-1)
    candidate_variance = np.asarray(candidate.variance, dtype=float).reshape(-1)
    if reference_mean.shape != candidate_mean.shape:
        raise ValueError("Posterior means must have matching shapes.")
    if reference_variance.shape != candidate_variance.shape:
        raise ValueError("Posterior variances must have matching shapes.")
    mean_error = candidate_mean - reference_mean
    variance_error = candidate_variance - reference_variance
    return {
        "mean_rmse": float(np.sqrt(np.mean(mean_error ** 2))),
        "mean_max_abs": float(np.max(np.abs(mean_error))),
        "variance_rmse": float(np.sqrt(np.mean(variance_error ** 2))),
        "variance_max_abs": float(np.max(np.abs(variance_error))),
    }


def gaussian_calibration_report(observed, mean, variance, levels=(0.9, 0.95)):
    """Return proper scores and interval calibration for Gaussian predictions."""
    observed = np.asarray(observed, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    variance = np.asarray(variance, dtype=float).reshape(-1)
    if observed.shape != mean.shape or observed.shape != variance.shape:
        raise ValueError("observed, mean, and variance must have matching shapes.")
    if not (np.all(np.isfinite(observed)) and np.all(np.isfinite(mean))):
        raise ValueError("observed and mean must be finite.")
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError("variance must be finite and positive.")
    levels = tuple(float(level) for level in levels)
    if any(level <= 0 or level >= 1 for level in levels):
        raise ValueError("calibration levels must lie strictly between 0 and 1.")

    standard_deviation = np.sqrt(variance)
    residual = observed - mean
    standardised = residual / standard_deviation
    crps = standard_deviation * (
        standardised * (2.0 * norm.cdf(standardised) - 1.0)
        + 2.0 * norm.pdf(standardised)
        - 1.0 / np.sqrt(np.pi)
    )
    report = {
        "count": int(observed.size),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "crps": float(np.mean(crps)),
        "nll": float(np.mean(
            0.5 * (np.log(2.0 * np.pi * variance) + residual ** 2 / variance))),
        "pit_mean": float(np.mean(norm.cdf(standardised))),
        "pit_variance": float(np.var(norm.cdf(standardised))),
    }
    for level in levels:
        critical = float(norm.ppf(0.5 + level / 2.0))
        covered = np.abs(residual) <= critical * standard_deviation
        key = "%g" % level
        report["coverage_" + key] = float(np.mean(covered))
        report["coverage_error_" + key] = float(abs(np.mean(covered) - level))
        report["interval_width_" + key] = float(
            np.mean(2.0 * critical * standard_deviation))
    return report


__all__ = [
    "GaussianEvidence",
    "IntervalEvidence",
    "SoftEvidence",
    "StreamingRankState",
    "StreamingPrecisionState",
    "VISTAOperator",
    "VISTAResult",
    "VecchiaGraph",
    "build_vecchia_graph",
    "compile_vista_operator",
    "gaussian_calibration_report",
    "gaussian_exactness_report",
]
