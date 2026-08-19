# -*- coding: utf-8 -*-
"""Compiled, sparse Gaussian BME prediction.

This module implements the closed-form Gaussian BME update as a reusable
linear operator.  Neighbour search and covariance solves are performed once
by :func:`compile_bme_operator`.  New hard values or Gaussian soft means can
then be predicted with one sparse matrix multiplication.

The fast path is exact for Gaussian general knowledge and Gaussian soft data
with fixed variances.  It intentionally does not approximate non-Gaussian
soft PDFs.  ``backend='auto'`` uses a CUDA batched Cholesky path for the
supported separable exponential space-time kernel and falls back to the
generic CPU implementation for every other model or neighbourhood shape.
"""

from __future__ import division

import time
import warnings

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from ..general.coord2K import coord2K
from ..general.isspacetime import isspacetime


def warmup_cuda():
    """Initialize CUDA once at service startup.

    Returns ``True`` when a CUDA device is ready and ``False`` when the
    optional PyTorch dependency or a CUDA device is unavailable.  Calling
    this during application startup keeps CUDA context initialization out of
    the first user prediction request.
    """
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    return True


class CompiledBMEOperator(object):
    """A reusable sparse operator for Gaussian BME posterior moments.

    Instances are created by :func:`compile_bme_operator`.  The observation
    vector is ordered as all hard values followed by all Gaussian soft means.
    """

    def __init__(self, weights, variance, valid, n_hard, n_soft, trend,
                 metadata=None, gpu_weights=None, gpu_indices=None,
                 gpu_chunk_size=2048, weight_shape=None):
        self._weights = None if weights is None else weights.tocsr()
        self._gpu_weights = gpu_weights
        self._gpu_indices = None if gpu_indices is None else np.asarray(
            gpu_indices, dtype=np.int64)
        if weight_shape is None:
            if self._weights is not None:
                weight_shape = self._weights.shape
            elif self._gpu_weights is not None:
                weight_shape = (
                    int(self._gpu_weights.shape[0]),
                    int(n_hard + n_soft),
                )
            else:
                raise ValueError("weights or gpu_weights must be supplied.")
        self._weight_shape = tuple(weight_shape)
        self._gpu_chunk_size = int(gpu_chunk_size)
        self.variance = np.asarray(variance, dtype=float)
        self.valid = np.asarray(valid, dtype=bool)
        self.n_hard = int(n_hard)
        self.n_soft = int(n_soft)
        self.trend = trend
        self.metadata = dict(metadata or {})

    @property
    def n_targets(self):
        return self._weight_shape[0]

    @property
    def n_observations(self):
        return self._weight_shape[1]

    @property
    def weights(self):
        """Return CPU CSR weights, materializing a CUDA operator if needed."""
        if self._weights is None:
            self._weights = self._materialize_cpu_weights()
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = None if value is None else value.tocsr()

    def _materialize_cpu_weights(self):
        if self._gpu_weights is None or self._gpu_indices is None:
            raise RuntimeError("The operator has no numerical weights.")
        data = self._gpu_weights.detach().cpu().numpy().reshape(-1)
        rows = np.repeat(
            np.arange(self.n_targets, dtype=np.int64),
            self._gpu_indices.shape[1],
        )
        columns = self._gpu_indices.reshape(-1)
        result = csr_matrix(
            (data, (rows, columns)), shape=self._weight_shape)
        result.sum_duplicates()
        result.eliminate_zeros()
        result.sort_indices()
        return result

    @property
    def storage_bytes(self):
        """Approximate bytes used by the numerical operator arrays."""
        if self._weights is None:
            weight_bytes = (
                self._gpu_weights.numel() * self._gpu_weights.element_size()
                + self._gpu_indices.nbytes
            )
            return int(weight_bytes + self.variance.nbytes + self.valid.nbytes)
        return int(
            self._weights.data.nbytes
            + self._weights.indices.nbytes
            + self._weights.indptr.nbytes
            + self.variance.nbytes
            + self.valid.nbytes
        )

    def predict(self, hard_values=None, soft_means=None):
        """Return posterior means for one or many observation updates.

        Parameters
        ----------
        hard_values : array-like, shape (n_hard,) or (n_hard, n_updates)
            Current hard observations.
        soft_means : array-like, shape (n_soft,) or (n_soft, n_updates)
            Current means of the Gaussian soft observations.  Their fixed
            variances were supplied when the operator was compiled.

        Returns
        -------
        ndarray
            Shape ``(n_targets,)`` for one-dimensional inputs, otherwise
            ``(n_targets, n_updates)``.
        """
        hard, hard_vector = _as_value_matrix(
            hard_values, self.n_hard, "hard_values")
        soft, soft_vector = _as_value_matrix(
            soft_means, self.n_soft, "soft_means")

        arrays = [a for a in (hard, soft) if a is not None]
        if not arrays:
            raise ValueError("At least one hard or soft observation is required.")

        update_count = arrays[0].shape[1]
        if any(a.shape[1] != update_count for a in arrays[1:]):
            raise ValueError(
                "hard_values and soft_means must have the same number of "
                "update columns."
            )

        observations = np.vstack(arrays)
        if self._gpu_weights is None:
            result = np.asarray(self._weights.dot(observations), dtype=float)
        else:
            result = self._predict_gpu(observations)
        result[~self.valid, :] = np.nan

        supplied_vector_flags = []
        if hard is not None:
            supplied_vector_flags.append(hard_vector)
        if soft is not None:
            supplied_vector_flags.append(soft_vector)
        if update_count == 1 and all(supplied_vector_flags):
            return result[:, 0]
        return result

    def _predict_gpu(self, observations):
        """Apply CUDA-resident weights while accepting ordinary NumPy input."""
        import torch

        device = self._gpu_weights.device
        values = torch.as_tensor(
            observations, dtype=torch.float64, device=device)
        output = []
        with torch.no_grad():
            for start in range(0, self.n_targets, self._gpu_chunk_size):
                stop = min(start + self._gpu_chunk_size, self.n_targets)
                indices = torch.as_tensor(
                    self._gpu_indices[start:stop],
                    dtype=torch.long,
                    device=device,
                )
                local_values = values[indices]
                output.append(torch.sum(
                    self._gpu_weights[start:stop, :, None] * local_values,
                    dim=1,
                ))
        return torch.cat(output, dim=0).cpu().numpy()

    def posterior_moments(self, hard_values=None, soft_means=None):
        """Return mean, variance, and skewness in STAMPS ``(n, 3)`` form."""
        mean = np.asarray(self.predict(hard_values, soft_means), dtype=float)
        if mean.ndim == 2:
            if mean.shape[1] != 1:
                raise ValueError(
                    "posterior_moments accepts one update. Use predict for "
                    "batched updates."
                )
            mean = mean[:, 0]
        skewness = np.zeros(self.n_targets, dtype=float)
        skewness[~self.valid] = np.nan
        return np.column_stack((mean, self.variance, skewness))

    def prediction_delta(self, hard_delta=None, soft_mean_delta=None):
        """Return the change in prediction for observation-value changes."""
        return self.predict(hard_delta, soft_mean_delta)

    def __repr__(self):
        return (
            "CompiledBMEOperator(n_targets={0}, n_hard={1}, n_soft={2}, "
            "nnz={3}, trend={4!r})"
        ).format(
            self.n_targets,
            self.n_hard,
            self.n_soft,
            self.weights.nnz,
            self.trend,
        )


def compile_bme_operator(
        ck, ch=None, cs=None, covmodel=None, covparam=None,
        nhmax=None, nsmax=None, dmax=None, soft_variance=None,
        hard_variance=0.0, trend="zero", workers=1,
        query_chunk_size=50000, jitter=1e-12, max_jitter_steps=6,
        backend="auto", gpu_chunk_size=2048):
    """Compile a sparse Gaussian BME influence operator.

    Parameters
    ----------
    ck, ch, cs : array-like
        Target, hard-data, and soft-data coordinates.  Three columns are
        interpreted as ``x, y, time``, matching the existing STAMPS BME API.
        Time coordinates must be numeric.
    covmodel, covparam : sequence
        Any covariance model accepted by :func:`stamps.general.coord2K`.
    nhmax, nsmax : int, optional
        Maximum hard and soft neighbours per target.  The defaults use every
        available observation of each type.
    dmax : scalar or sequence ``(space, time, ratio)``, optional
        Search radius.  Space-time searches use the same normalized Euclidean
        radius as ``BMEPosteriorMoments``.  With no value, nearest-neighbour
        selection has no distance cutoff.
    soft_variance : scalar or array-like
        Fixed Gaussian soft-data variances.  Required when ``cs`` is present.
    hard_variance : scalar or array-like, default 0
        Optional hard-observation error variances.
    trend : {"zero", "local_constant"}, default "zero"
        ``zero`` matches ``order=np.nan``.  ``local_constant`` matches the
        local observation average used by ``order=0``.
    workers : int, default 1
        Workers used by SciPy's cKDTree queries.  Use ``-1`` for all cores.
    query_chunk_size : int, default 50000
        Bounds temporary neighbour-query memory for large target sets.
    jitter : float, default 1e-12
        Initial diagonal jitter relative to the covariance diagonal scale,
        used only if an unmodified Cholesky factorization fails.
    max_jitter_steps : int, default 6
        Number of tenfold jitter increases before a pseudoinverse fallback.
    backend : {"auto", "cpu", "gpu"}, default "auto"
        ``auto`` selects the CUDA path for supported large local problems and
        otherwise uses the generic CPU implementation.  ``gpu`` requires a
        CUDA-enabled PyTorch installation and raises if the problem is not
        supported by the exact batched path.
    gpu_chunk_size : int, default 2048
        Number of targets processed by one CUDA batch.

    Returns
    -------
    CompiledBMEOperator
        Reusable posterior-mean weights and posterior marginal variances.

    Notes
    -----
    This is the closed-form Gaussian update

    ``K_ko @ inv(K_oo + R) @ y``.

    Gaussian soft data enter as noisy observations whose error covariance is
    their supplied diagonal variance.  Changing coordinates, covariance
    parameters, neighbourhood settings, or observation variances requires a
    recompile.  Changing only hard values or soft means does not.
    """
    started = time.perf_counter()

    if covmodel is None or covparam is None:
        raise ValueError("covmodel and covparam are required.")
    if isinstance(covmodel, str) or not hasattr(covmodel, "__len__"):
        raise TypeError("covmodel must be a non-empty sequence of model names.")
    if not hasattr(covparam, "__len__"):
        raise TypeError("covparam must be a sequence of parameter records.")
    if len(covmodel) == 0 or len(covparam) == 0:
        raise ValueError("covmodel and covparam cannot be empty.")
    if len(covmodel) != len(covparam):
        raise ValueError("covmodel and covparam must have the same length.")

    targets = _as_coordinates(ck, "ck", allow_empty=False)
    hard_coords = _as_optional_coordinates(ch, targets.shape[1], "ch")
    soft_coords = _as_optional_coordinates(cs, targets.shape[1], "cs")
    n_targets = targets.shape[0]
    n_hard = hard_coords.shape[0]
    n_soft = soft_coords.shape[0]
    if n_hard + n_soft == 0:
        raise ValueError("Hard and soft coordinates cannot both be empty.")

    hard_limit = _validate_neighbour_count(nhmax, n_hard, "nhmax")
    soft_limit = _validate_neighbour_count(nsmax, n_soft, "nsmax")
    if hard_limit + soft_limit == 0:
        raise ValueError("nhmax and nsmax cannot both select zero neighbours.")

    hard_var = _broadcast_variance(
        hard_variance, n_hard, "hard_variance", required=False)
    soft_var = _broadcast_variance(
        soft_variance, n_soft, "soft_variance", required=(n_soft > 0))
    trend = _normalise_trend(trend)

    if workers is not None:
        if not isinstance(workers, (int, np.integer)):
            raise TypeError("workers must be an integer or None.")
        if workers == 0 or workers < -1:
            raise ValueError("workers must be -1 or a positive integer.")
    if not isinstance(query_chunk_size, (int, np.integer)):
        raise TypeError("query_chunk_size must be an integer.")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be greater than zero.")
    if not np.isscalar(jitter) or not np.isfinite(jitter) or jitter < 0:
        raise ValueError("jitter must be a finite non-negative scalar.")
    if not isinstance(max_jitter_steps, (int, np.integer)):
        raise TypeError("max_jitter_steps must be an integer.")
    if max_jitter_steps < 0:
        raise ValueError("max_jitter_steps must be non-negative.")
    backend = _normalise_backend(backend)
    if not isinstance(gpu_chunk_size, (int, np.integer)):
        raise TypeError("gpu_chunk_size must be an integer.")
    if gpu_chunk_size <= 0:
        raise ValueError("gpu_chunk_size must be greater than zero.")

    target_search, hard_search, soft_search, search_radius, time_scale = (
        _prepare_search_coordinates(
            targets, hard_coords, soft_coords, dmax, covmodel, covparam)
    )
    hard_tree = cKDTree(hard_search) if hard_limit else None
    soft_tree = cKDTree(soft_search) if soft_limit else None

    should_try_gpu = (
        backend == "gpu"
        or (backend == "auto" and targets.shape[1] == 3
            and n_targets >= 8192 and hard_limit + soft_limit <= 128)
    )
    if should_try_gpu:
        try:
            gpu_operator = _compile_gpu_operator(
                targets=targets,
                hard_coords=hard_coords,
                soft_coords=soft_coords,
                hard_var=hard_var,
                soft_var=soft_var,
                hard_limit=hard_limit,
                soft_limit=soft_limit,
                covmodel=covmodel,
                covparam=covparam,
                trend=trend,
                target_search=target_search,
                hard_tree=hard_tree,
                soft_tree=soft_tree,
                search_radius=search_radius,
                workers=workers,
                jitter=jitter,
                max_jitter_steps=max_jitter_steps,
                gpu_chunk_size=gpu_chunk_size,
                started=started,
            )
        except Exception as error:
            if backend == "gpu":
                raise RuntimeError(
                    "The CUDA BME backend failed; use backend='auto' "
                    "to allow a CPU fallback."
                ) from error
            warnings.warn(
                "CUDA BME backend failed; falling back to the CPU path: "
                "{0}".format(error),
                RuntimeWarning,
            )
            gpu_operator = None
        if gpu_operator is not None:
            return gpu_operator
        if backend == "gpu":
            raise RuntimeError(
                "backend='gpu' only supports a CUDA device, a single "
                "separable exponentialC/exponentialC model, and complete "
                "fixed-size local neighbourhoods."
            )

    groups = {}
    for start in range(0, n_targets, query_chunk_size):
        stop = min(start + query_chunk_size, n_targets)
        target_chunk = target_search[start:stop]
        hard_neighbours = _query_neighbour_tuples(
            hard_tree, target_chunk, hard_limit, search_radius, workers)
        soft_neighbours = _query_neighbour_tuples(
            soft_tree, target_chunk, soft_limit, search_radius, workers)
        for local_index, pair in enumerate(zip(
                hard_neighbours, soft_neighbours)):
            if not pair[0] and not pair[1]:
                continue
            groups.setdefault(pair, []).append(start + local_index)

    nonzero_count = sum(
        len(target_indices) * (len(key[0]) + len(key[1]))
        for key, target_indices in groups.items()
    )
    row_indices = np.empty(nonzero_count, dtype=np.int64)
    column_indices = np.empty(nonzero_count, dtype=np.int64)
    weight_data = np.empty(nonzero_count, dtype=float)
    posterior_variance = np.full(n_targets, np.nan, dtype=float)
    valid = np.zeros(n_targets, dtype=bool)

    prior_variance = float(
        coord2K(targets[:1], targets[:1], covmodel, covparam)[0][0, 0]
    )
    if not np.isfinite(prior_variance) or prior_variance < 0:
        raise ValueError(
            "The covariance model produced an invalid marginal variance."
        )

    write_position = 0
    jittered_groups = 0
    pseudoinverse_groups = 0
    max_jitter_used = 0.0
    negative_variance_count = 0

    for (hard_tuple, soft_tuple), target_index_list in groups.items():
        hard_index = np.asarray(hard_tuple, dtype=int)
        soft_index = np.asarray(soft_tuple, dtype=int)
        target_index = np.asarray(target_index_list, dtype=int)

        coordinate_parts = []
        variance_parts = []
        observation_columns = []
        if hard_index.size:
            coordinate_parts.append(hard_coords[hard_index])
            variance_parts.append(hard_var[hard_index])
            observation_columns.append(hard_index)
        if soft_index.size:
            coordinate_parts.append(soft_coords[soft_index])
            variance_parts.append(soft_var[soft_index])
            observation_columns.append(n_hard + soft_index)

        observation_coords = np.vstack(coordinate_parts)
        observation_variance = np.concatenate(variance_parts)
        observation_columns = np.concatenate(observation_columns)

        covariance_oo = coord2K(
            observation_coords, observation_coords, covmodel, covparam)[0]
        covariance_oo = 0.5 * (covariance_oo + covariance_oo.T)
        covariance_oo = np.asarray(covariance_oo, dtype=float)
        covariance_oo.flat[::covariance_oo.shape[0] + 1] += observation_variance
        covariance_ok = coord2K(
            observation_coords, targets[target_index], covmodel, covparam)[0]
        if not (np.all(np.isfinite(covariance_oo))
                and np.all(np.isfinite(covariance_ok))):
            raise ValueError("The covariance model produced non-finite values.")

        raw_weights, jitter_used, used_pseudoinverse = _solve_covariance(
            covariance_oo, covariance_ok, jitter, max_jitter_steps)
        if jitter_used:
            jittered_groups += 1
            max_jitter_used = max(max_jitter_used, jitter_used)
        if used_pseudoinverse:
            pseudoinverse_groups += 1

        group_variance = prior_variance - np.einsum(
            "ij,ij->j", covariance_ok, raw_weights)
        numerical_tolerance = 1e-10 * max(1.0, abs(prior_variance))
        small_negative = (
            (group_variance < 0.0)
            & (group_variance >= -numerical_tolerance)
        )
        group_variance[small_negative] = 0.0
        negative_variance_count += int(np.count_nonzero(
            group_variance < -numerical_tolerance))

        operator_weights = raw_weights
        if trend == "local_constant":
            observation_count = raw_weights.shape[0]
            correction = (
                1.0 - np.sum(raw_weights, axis=0, keepdims=True)
            ) / observation_count
            operator_weights = raw_weights + correction

        rows_in_group = target_index.size
        observations_in_group = observation_columns.size
        block_size = rows_in_group * observations_in_group
        block_slice = slice(write_position, write_position + block_size)
        row_indices[block_slice] = np.repeat(
            target_index, observations_in_group)
        column_indices[block_slice] = np.tile(
            observation_columns, rows_in_group)
        weight_data[block_slice] = operator_weights.T.reshape(-1)
        write_position += block_size

        posterior_variance[target_index] = group_variance
        valid[target_index] = True

    weights = csr_matrix(
        (weight_data, (row_indices, column_indices)),
        shape=(n_targets, n_hard + n_soft),
    )
    weights.sum_duplicates()
    weights.eliminate_zeros()
    weights.sort_indices()

    if jittered_groups:
        warnings.warn(
            "Cholesky factorization required diagonal jitter in {0} "
            "neighbour group(s); maximum absolute jitter was {1:.3g}."
            .format(jittered_groups, max_jitter_used),
            RuntimeWarning,
        )
    if pseudoinverse_groups:
        warnings.warn(
            "Cholesky factorization failed after jitter in {0} neighbour "
            "group(s); a pseudoinverse was used.".format(
                pseudoinverse_groups),
            RuntimeWarning,
        )
    if negative_variance_count:
        warnings.warn(
            "The covariance specification produced {0} materially negative "
            "posterior variance value(s).".format(negative_variance_count),
            RuntimeWarning,
        )

    metadata = {
        "compile_seconds": time.perf_counter() - started,
        "backend": "cpu",
        "device": "cpu",
        "group_count": len(groups),
        "valid_target_count": int(np.count_nonzero(valid)),
        "jittered_group_count": jittered_groups,
        "pseudoinverse_group_count": pseudoinverse_groups,
        "max_jitter_used": max_jitter_used,
        "time_scale": time_scale,
        "search_radius": search_radius,
        "prior_variance": prior_variance,
    }
    return CompiledBMEOperator(
        weights=weights,
        variance=posterior_variance,
        valid=valid,
        n_hard=n_hard,
        n_soft=n_soft,
        trend=trend,
        metadata=metadata,
    )


def _normalise_backend(backend):
    if not isinstance(backend, str):
        raise TypeError("backend must be 'auto', 'cpu', or 'gpu'.")
    normalised = backend.strip().lower()
    if normalised not in ("auto", "cpu", "gpu"):
        raise ValueError("backend must be 'auto', 'cpu', or 'gpu'.")
    return normalised


def _compile_gpu_operator(
        targets, hard_coords, soft_coords, hard_var, soft_var,
        hard_limit, soft_limit, covmodel, covparam, trend,
        target_search, hard_tree, soft_tree, search_radius, workers,
        jitter, max_jitter_steps, gpu_chunk_size, started):
    """Compile the exact local operator with CUDA batched Cholesky.

    This deliberately supports a narrow, well-tested model family.  A
    generic GPU covariance dispatcher would make ``backend='auto'`` less
    predictable and could silently change numerical results.  Unsupported
    inputs return ``None`` so the caller can use the mature CPU path.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    if targets.shape[1] != 3:
        return None
    if len(covmodel) != 1 or len(covparam) != 1:
        return None
    if str(covmodel[0]).strip().lower() != "exponentialc/exponentialc":
        return None
    if hard_limit + soft_limit == 0:
        return None

    try:
        sill = float(np.asarray(covparam[0][0], dtype=float).reshape(-1)[0])
        space_range = _first_positive_number(covparam[0][1])
        time_range = _first_positive_number(covparam[0][2])
    except (IndexError, TypeError, ValueError):
        return None
    if not np.isfinite(sill) or sill < 0:
        return None

    hard_neighbours = _query_neighbour_indices(
        hard_tree, target_search, hard_limit, search_radius, workers)
    soft_neighbours = _query_neighbour_indices(
        soft_tree, target_search, soft_limit, search_radius, workers)
    if hard_neighbours is None or soft_neighbours is None:
        return None

    index_parts = []
    if hard_limit:
        index_parts.append(hard_neighbours)
    if soft_limit:
        index_parts.append(soft_neighbours + hard_coords.shape[0])
    observation_indices = np.concatenate(index_parts, axis=1)
    observation_coords = np.concatenate((hard_coords, soft_coords), axis=0)
    observation_variance = np.concatenate((hard_var, soft_var), axis=0)

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    dtype = torch.float64
    target_tensor = torch.as_tensor(targets, dtype=dtype, device=device)
    coordinate_tensor = torch.as_tensor(
        observation_coords, dtype=dtype, device=device)
    variance_tensor = torch.as_tensor(
        observation_variance, dtype=dtype, device=device)

    all_weights = []
    all_variances = []
    jittered_batches = 0
    max_jitter_used = 0.0
    observation_count = observation_indices.shape[1]

    for start in range(0, targets.shape[0], gpu_chunk_size):
        stop = min(start + gpu_chunk_size, targets.shape[0])
        indices = torch.as_tensor(
            observation_indices[start:stop], dtype=torch.long, device=device)
        local_coords = coordinate_tensor[indices]
        local_variance = variance_tensor[indices]
        local_space = local_coords[:, :, :2]
        local_time = local_coords[:, :, 2]
        space_distance = torch.linalg.vector_norm(
            local_space[:, :, None, :] - local_space[:, None, :, :],
            dim=3,
        )
        time_distance = torch.abs(
            local_time[:, :, None] - local_time[:, None, :])
        covariance = sill * torch.exp(
            -3.0 * space_distance / space_range
            -3.0 * time_distance / time_range
        )
        covariance.diagonal(dim1=1, dim2=2).add_(local_variance)

        local_targets = target_tensor[start:stop]
        target_space_distance = torch.linalg.vector_norm(
            local_targets[:, None, :2] - local_space, dim=2)
        target_time_distance = torch.abs(
            local_targets[:, 2, None] - local_time)
        covariance_ok = sill * torch.exp(
            -3.0 * target_space_distance / space_range
            -3.0 * target_time_distance / time_range
        )

        factor, info = torch.linalg.cholesky_ex(
            covariance, check_errors=False)
        jitter_used = 0.0
        if torch.any(info != 0):
            diagonal_scale = max(
                1.0, sill + float(np.max(observation_variance))
            )
            jitter_used = jitter * diagonal_scale
            regularised = covariance
            for unused in range(max_jitter_steps):
                if jitter_used <= 0:
                    break
                regularised = covariance.clone()
                regularised.diagonal(dim1=1, dim2=2).add_(jitter_used)
                factor, info = torch.linalg.cholesky_ex(
                    regularised, check_errors=False)
                if not torch.any(info != 0):
                    break
                jitter_used *= 10.0
            if torch.any(info != 0):
                return None
            jittered_batches += 1
            max_jitter_used = max(max_jitter_used, jitter_used)

        weights = torch.cholesky_solve(
            covariance_ok[:, :, None], factor).squeeze(2)
        if trend == "local_constant":
            correction = (
                1.0 - torch.sum(weights, dim=1, keepdim=True)
            ) / observation_count
            weights = weights + correction
        posterior_variance = sill - torch.sum(
            covariance_ok * weights, dim=1)

        tolerance = 1e-10 * max(1.0, abs(sill))
        posterior_variance = torch.where(
            (posterior_variance < 0.0)
            & (posterior_variance >= -tolerance),
            torch.zeros_like(posterior_variance),
            posterior_variance,
        )
        all_weights.append(weights)
        all_variances.append(posterior_variance)

    gpu_weights = torch.cat(all_weights, dim=0)
    posterior_variance = torch.cat(all_variances, dim=0).cpu().numpy()
    valid = np.ones(targets.shape[0], dtype=bool)

    if jittered_batches:
        warnings.warn(
            "CUDA Cholesky required diagonal jitter in {0} batch(es); "
            "maximum absolute jitter was {1:.3g}.".format(
                jittered_batches, max_jitter_used),
            RuntimeWarning,
        )
    metadata = {
        "compile_seconds": time.perf_counter() - started,
        "backend": "gpu",
        "device": "cuda",
        "dtype": "float64",
        "group_count": int(targets.shape[0]),
        "valid_target_count": int(targets.shape[0]),
        "jittered_batch_count": jittered_batches,
        "max_jitter_used": max_jitter_used,
        "time_scale": _infer_time_scale(covmodel, covparam),
        "search_radius": search_radius,
        "prior_variance": sill,
        "gpu_chunk_size": gpu_chunk_size,
    }
    return CompiledBMEOperator(
        weights=None,
        variance=posterior_variance,
        valid=valid,
        n_hard=hard_coords.shape[0],
        n_soft=soft_coords.shape[0],
        trend=trend,
        metadata=metadata,
        gpu_weights=gpu_weights,
        gpu_indices=observation_indices,
        gpu_chunk_size=gpu_chunk_size,
        weight_shape=(
            targets.shape[0], hard_coords.shape[0] + soft_coords.shape[0]),
        
    )


def _as_coordinates(values, label, allow_empty):
    if values is None:
        raise ValueError("{0} is required.".format(label))
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.datetime64):
        raise TypeError(
            "{0} must use numeric coordinates; convert datetime values to "
            "elapsed numeric time first.".format(label)
        )
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("{0} must contain numeric coordinates.".format(label))
    if array.ndim != 2:
        raise ValueError("{0} must be a two-dimensional array.".format(label))
    if array.shape[1] == 0:
        raise ValueError("{0} must have at least one coordinate column.".format(label))
    if not allow_empty and array.shape[0] == 0:
        raise ValueError("{0} cannot be empty.".format(label))
    if not np.all(np.isfinite(array)):
        raise ValueError("{0} contains NaN or infinite coordinates.".format(label))
    return np.ascontiguousarray(array)


def _as_optional_coordinates(values, dimension, label):
    if values is None:
        return np.empty((0, dimension), dtype=float)
    array = _as_coordinates(values, label, allow_empty=True)
    if array.shape[1] != dimension:
        raise ValueError(
            "{0} must have {1} coordinate column(s), matching ck."
            .format(label, dimension)
        )
    return array


def _validate_neighbour_count(value, available, label):
    if value is None:
        return int(available)
    if not isinstance(value, (int, np.integer)):
        raise TypeError("{0} must be an integer or None.".format(label))
    if value < 0:
        raise ValueError("{0} must be non-negative.".format(label))
    return int(min(value, available))


def _broadcast_variance(values, size, label, required):
    if size == 0:
        if values is not None and np.asarray(values).size not in (0, 1):
            raise ValueError("{0} was supplied without coordinates.".format(label))
        return np.empty(0, dtype=float)
    if values is None:
        if required:
            raise ValueError("{0} is required when cs is present.".format(label))
        values = 0.0
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=float)
    else:
        array = array.reshape(-1)
        if array.size != size:
            raise ValueError(
                "{0} must be scalar or contain exactly {1} value(s)."
                .format(label, size)
            )
    if not np.all(np.isfinite(array)):
        raise ValueError("{0} contains NaN or infinite values.".format(label))
    if np.any(array < 0.0):
        raise ValueError("{0} cannot contain negative variances.".format(label))
    return array


def _normalise_trend(trend):
    if isinstance(trend, str):
        normalised = trend.strip().lower().replace("-", "_")
        aliases = {
            "zero": "zero",
            "zero_mean": "zero",
            "local_constant": "local_constant",
            "constant": "local_constant",
        }
        if normalised in aliases:
            return aliases[normalised]
    elif np.isscalar(trend):
        if np.isnan(trend):
            return "zero"
        if trend == 0:
            return "local_constant"
    raise ValueError(
        "trend must be 'zero', 'local_constant', np.nan, or 0."
    )


def _prepare_search_coordinates(
        targets, hard_coords, soft_coords, dmax, covmodel, covparam):
    dimension = targets.shape[1]
    search_radius = None
    time_scale = 1.0

    if dimension == 3:
        dmax_values = None if dmax is None else np.asarray(
            dmax, dtype=float).reshape(-1)
        if dmax_values is not None and dmax_values.size != 3:
            raise ValueError(
                "For x, y, time coordinates, dmax must contain "
                "(space, time, ratio)."
            )
        if dmax_values is not None and (
                not np.all(np.isfinite(dmax_values[:2]))
                or dmax_values[0] < 0 or dmax_values[1] < 0):
            raise ValueError(
                "dmax space and time limits must be finite and non-negative."
            )

        if dmax_values is None or np.isnan(dmax_values[2]):
            time_scale = _infer_time_scale(covmodel, covparam)
        else:
            time_scale = float(dmax_values[2])
        if not np.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("The space-time ratio must be finite and positive.")

        if dmax_values is not None:
            search_radius = float(np.hypot(
                dmax_values[0], dmax_values[1] * time_scale))

        transformed = []
        for coordinates in (targets, hard_coords, soft_coords):
            current = np.array(coordinates, dtype=float, copy=True)
            current[:, 2] *= time_scale
            transformed.append(current)
        return tuple(transformed) + (search_radius, time_scale)

    if dmax is not None:
        dmax_values = np.asarray(dmax, dtype=float).reshape(-1)
        if dmax_values.size != 1:
            raise ValueError(
                "For non-space-time coordinates, dmax must be a scalar."
            )
        search_radius = float(dmax_values[0])
        if not np.isfinite(search_radius) or search_radius < 0:
            raise ValueError("dmax must be finite and non-negative.")
    return (
        targets.copy(), hard_coords.copy(), soft_coords.copy(),
        search_radius, time_scale,
    )


def _infer_time_scale(covmodel, covparam):
    try:
        is_space_time, is_separable, unused = isspacetime(covmodel)
    except Exception:
        is_space_time = False
        is_separable = False
    if not is_space_time:
        return 1.0

    if len(covparam) == 0:
        raise ValueError(
            "Cannot infer the space-time ratio from empty covparam."
        )
    sills = []
    for parameter in covparam:
        try:
            sills.append(float(np.asarray(parameter[0]).reshape(-1)[0]))
        except (IndexError, TypeError, ValueError):
            sills.append(-np.inf)
    component_index = int(np.argmax(sills))
    parameter = covparam[component_index]

    if is_separable:
        try:
            spatial_range = _first_positive_number(parameter[1])
            temporal_range = _first_positive_number(parameter[2])
            return spatial_range / temporal_range
        except (IndexError, TypeError, ValueError):
            raise ValueError(
                "Cannot infer a space-time ratio from covparam. Supply "
                "dmax=(space, time, ratio) explicitly."
            )

    try:
        return _first_positive_number(parameter[2])
    except (IndexError, TypeError, ValueError):
        raise ValueError(
            "Cannot infer a space-time ratio from covparam. Supply "
            "dmax=(space, time, ratio) explicitly."
        )


def _first_positive_number(value):
    array = np.asarray(value, dtype=float).reshape(-1)
    candidates = array[np.isfinite(array) & (array > 0)]
    if candidates.size == 0:
        raise ValueError("No positive finite covariance parameter was found.")
    return float(candidates[0])


def _query_neighbour_tuples(tree, targets, neighbour_count, radius, workers):
    if tree is None or neighbour_count == 0:
        return [()] * targets.shape[0]
    distance_upper_bound = np.inf if radius is None else radius
    query_arguments = {
        "k": neighbour_count,
        "distance_upper_bound": distance_upper_bound,
    }
    if workers is not None:
        query_arguments["workers"] = workers
    try:
        distances, indices = tree.query(targets, **query_arguments)
    except TypeError:
        query_arguments.pop("workers", None)
        distances, indices = tree.query(targets, **query_arguments)

    distances = np.asarray(distances)
    indices = np.asarray(indices)
    if neighbour_count == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    result = []
    tree_size = tree.n
    for distance_row, index_row in zip(distances, indices):
        valid = np.isfinite(distance_row) & (index_row < tree_size)
        result.append(tuple(sorted(index_row[valid].astype(int).tolist())))
    return result


def _query_neighbour_indices(tree, targets, neighbour_count, radius, workers):
    """Return a dense neighbour-index matrix without Python row loops."""
    if tree is None or neighbour_count == 0:
        return np.empty((targets.shape[0], 0), dtype=np.int64)
    distance_upper_bound = np.inf if radius is None else radius
    query_arguments = {
        "k": neighbour_count,
        "distance_upper_bound": distance_upper_bound,
    }
    if workers is not None:
        query_arguments["workers"] = workers
    try:
        distances, indices = tree.query(targets, **query_arguments)
    except TypeError:
        query_arguments.pop("workers", None)
        distances, indices = tree.query(targets, **query_arguments)
    distances = np.asarray(distances)
    indices = np.asarray(indices)
    if neighbour_count == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)
    valid = np.isfinite(distances) & (indices < tree.n)
    if not np.all(valid):
        return None
    return np.ascontiguousarray(indices, dtype=np.int64)


def _solve_covariance(matrix, right_hand_side, relative_jitter,
                      max_jitter_steps):
    try:
        factor = cho_factor(matrix, lower=True, check_finite=False)
        return (
            cho_solve(factor, right_hand_side, check_finite=False),
            0.0,
            False,
        )
    except LinAlgError:
        pass

    diagonal_scale = max(
        1.0, float(np.max(np.abs(np.diag(matrix))))
    )
    absolute_jitter = relative_jitter * diagonal_scale
    if absolute_jitter > 0:
        for unused in range(max_jitter_steps):
            regularised = matrix.copy()
            regularised.flat[::regularised.shape[0] + 1] += absolute_jitter
            try:
                factor = cho_factor(
                    regularised, lower=True, check_finite=False)
                return (
                    cho_solve(
                        factor, right_hand_side, check_finite=False),
                    absolute_jitter,
                    False,
                )
            except LinAlgError:
                absolute_jitter *= 10.0

    try:
        inverse = np.linalg.pinv(matrix, hermitian=True)
    except TypeError:
        inverse = np.linalg.pinv(matrix)
    return inverse.dot(right_hand_side), 0.0, True


def _as_value_matrix(values, expected_size, label):
    if expected_size == 0:
        if values is not None and np.asarray(values).size:
            raise ValueError(
                "{0} was supplied but the operator has no matching "
                "coordinates.".format(label)
            )
        return None, True
    if values is None:
        raise ValueError("{0} is required.".format(label))
    array = np.asarray(values, dtype=float)
    was_vector = array.ndim == 1
    if array.ndim == 1:
        if array.shape[0] != expected_size:
            raise ValueError(
                "{0} must contain {1} value(s).".format(
                    label, expected_size)
            )
        array = array.reshape(expected_size, 1)
    elif array.ndim == 2:
        if array.shape[0] != expected_size:
            raise ValueError(
                "{0} must have {1} row(s).".format(label, expected_size)
            )
        if array.shape[1] == 0:
            raise ValueError("{0} cannot have zero update columns.".format(label))
    else:
        raise ValueError(
            "{0} must be one- or two-dimensional.".format(label)
        )
    return np.ascontiguousarray(array), was_vector


FastBME = CompiledBMEOperator


__all__ = [
    "CompiledBMEOperator",
    "FastBME",
    "compile_bme_operator",
    "warmup_cuda",
]
