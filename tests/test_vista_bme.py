# -*- coding: utf-8 -*-

from __future__ import division

import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from stamps.bme import (
    CensoredEvidence,
    GaussianEvidence,
    GaussianMixtureEvidence,
    IntervalEvidence,
    LogNormalEvidence,
    SkewNormalEvidence,
    StreamingPrecisionState,
    StreamingRankState,
    StreamingSiteAccumulator,
    StudentTEvidence,
    TabulatedEvidence,
    build_vecchia_graph,
    compile_vista_operator,
    evidence_pdf,
    gaussian_calibration_report,
    gaussian_exactness_report,
)
from stamps.general.coord2K import coord2K


class VISTABMETest(unittest.TestCase):

    def setUp(self):
        self.ck = np.array([
            [0.15, 0.20],
            [0.45, 0.35],
            [0.80, 0.25],
            [0.70, 0.85],
        ])
        self.ch = np.array([
            [0.00, 0.00],
            [1.00, 0.00],
            [0.00, 1.00],
            [1.00, 1.00],
            [0.50, 0.55],
        ])
        self.hard_values = np.array([1.0, 2.0, 3.1, 3.9, 2.6])
        self.cs = np.array([
            [0.25, 0.60],
            [0.65, 0.15],
            [0.85, 0.70],
        ])
        soft_mean = (2.35, 1.95, 3.55)
        soft_variance = (0.12, 0.20, 0.08)
        self.soft_evidence = tuple(
            GaussianEvidence(mean, variance)
            for mean, variance in zip(soft_mean, soft_variance)
        )
        self.covmodel = ["exponentialC"]
        self.covparam = [(1.7, [1.25])]

    def _operator(self, mode, max_parents):
        return compile_vista_operator(
            self.ck,
            ch=self.ch,
            cs=self.cs,
            soft_evidence=self.soft_evidence,
            covmodel=self.covmodel,
            covparam=self.covparam,
            mode=mode,
            max_parents=max_parents,
            ordering="random",
            random_state=17,
            ep_tolerance=1e-10,
        )

    def test_full_parent_precision_reconstructs_dense_covariance(self):
        coordinates = np.vstack((self.ch, self.cs, self.ck))
        covariance = np.asarray(coord2K(
            coordinates, coordinates, self.covmodel, self.covparam)[0],
            dtype=float,
        )
        graph = build_vecchia_graph(
            coordinates,
            self.covmodel,
            self.covparam,
            max_parents=None,
            ordering="random",
            random_state=31,
        )

        reconstructed = np.linalg.inv(graph.precision.toarray())
        np.testing.assert_allclose(
            reconstructed, covariance, rtol=2e-11, atol=2e-11)
        self.assertEqual(graph.metadata["stabilised_nodes"], 0)

    def test_full_parent_gaussian_matches_exact_bme(self):
        reference = self._operator("exact", None).predict(
            self.hard_values)
        candidate = self._operator("vecchia", None).predict(
            self.hard_values)

        np.testing.assert_allclose(
            candidate.mean, reference.mean, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(
            candidate.variance, reference.variance, rtol=2e-10, atol=2e-10)
        errors = gaussian_exactness_report(reference, candidate)
        self.assertLess(errors["mean_max_abs"], 2e-10)
        self.assertLess(errors["variance_max_abs"], 2e-10)
        self.assertTrue(candidate.metadata["exact_limit"])

    def test_sparse_gaussian_mode_returns_finite_moments(self):
        result = self._operator("vecchia", 3).predict(self.hard_values)

        self.assertTrue(np.all(np.isfinite(result.mean)))
        self.assertTrue(np.all(result.variance > 0))
        self.assertEqual(result.metadata["max_parents"], 3)
        self.assertLess(
            self._operator("vecchia", 3).graph.precision.nnz,
            self._operator("vecchia", None).graph.precision.nnz,
        )

    def test_one_interval_ep_matches_exact_first_two_moments(self):
        soft_coordinate = np.array([[0.0, 0.0]])
        target_coordinate = np.array([[0.55, 0.25]])
        interval = IntervalEvidence(-0.35, 0.80)
        operator = compile_vista_operator(
            target_coordinate,
            ch=np.empty((0, 2)),
            cs=soft_coordinate,
            soft_evidence=[interval],
            covmodel=self.covmodel,
            covparam=self.covparam,
            mode="vecchia",
            max_parents=None,
            ordering="input",
            ep_damping=1.0,
            ep_tolerance=1e-12,
            ep_max_iterations=10,
        )
        result = operator.predict([])

        covariance = np.asarray(coord2K(
            np.vstack((soft_coordinate, target_coordinate)),
            np.vstack((soft_coordinate, target_coordinate)),
            self.covmodel,
            self.covparam,
        )[0], dtype=float)
        soft_variance = covariance[0, 0]
        alpha = interval.lower / np.sqrt(soft_variance)
        beta = interval.upper / np.sqrt(soft_variance)
        probability = norm.cdf(beta) - norm.cdf(alpha)
        ratio = (norm.pdf(alpha) - norm.pdf(beta)) / probability
        truncated_mean = np.sqrt(soft_variance) * ratio
        truncated_variance = soft_variance * (
            1.0
            + (alpha * norm.pdf(alpha) - beta * norm.pdf(beta)) / probability
            - ratio ** 2
        )
        regression = covariance[1, 0] / soft_variance
        expected_mean = regression * truncated_mean
        expected_variance = (
            covariance[1, 1]
            - covariance[1, 0] ** 2 / soft_variance
            + regression ** 2 * truncated_variance
        )

        np.testing.assert_allclose(result.mean, [expected_mean], rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(
            result.variance, [expected_variance], rtol=2e-10, atol=2e-10)
        self.assertTrue(result.metadata["ep_converged"])
        self.assertLess(result.metadata["ep_standardized_mean_residual"], 1e-10)
        self.assertLess(result.metadata["ep_relative_variance_residual"], 1e-10)

    def test_interval_ep_remains_finite_in_far_gaussian_tail(self):
        operator = compile_vista_operator(
            np.array([[0.2, 0.0]]),
            ch=np.empty((0, 2)),
            cs=np.array([[0.0, 0.0]]),
            soft_evidence=[IntervalEvidence(9.0, 10.0)],
            covmodel=["exponentialC"],
            covparam=[(1.0, [1.0])],
            mode="vecchia",
            max_parents=None,
            ordering="input",
            ep_damping=1.0,
            ep_tolerance=1e-10,
            ep_max_iterations=20,
        )
        result = operator.predict([])

        self.assertTrue(np.all(np.isfinite(result.mean)))
        self.assertTrue(np.all(np.isfinite(result.variance)))
        self.assertGreater(result.variance[0], 0.0)

    def test_generalized_soft_evidence_matches_one_site_quadrature_moments(self):
        grid = np.linspace(-2.5, 3.5, 241)
        tabulated_density = np.exp(-0.5 * ((grid - 0.8) / 0.55) ** 2)
        tabulated_density *= 1.0 + 0.25 * np.sin(4.0 * grid) ** 2
        cases = (
            CensoredEvidence(lower=-0.4, upper=np.inf),
            SkewNormalEvidence(location=0.3, scale=0.8, shape=5.0),
            LogNormalEvidence(log_mean=-0.1, log_scale=0.45),
            StudentTEvidence(location=0.2, scale=0.55, degrees_of_freedom=4.0),
            TabulatedEvidence(grid, tabulated_density),
        )
        soft_coordinate = np.array([[0.0, 0.0]])
        target_coordinate = np.array([[0.35, 0.15]])
        coordinates = np.vstack((soft_coordinate, target_coordinate))
        covariance = np.asarray(coord2K(
            coordinates, coordinates, self.covmodel, self.covparam)[0], dtype=float)
        soft_variance = covariance[0, 0]
        regression = covariance[1, 0] / soft_variance
        conditional_variance = (
            covariance[1, 1] - covariance[1, 0] ** 2 / soft_variance)
        bound = 12.0 * np.sqrt(soft_variance)

        for evidence in cases:
            with self.subTest(evidence=type(evidence).__name__):
                operator = compile_vista_operator(
                    target_coordinate,
                    ch=np.empty((0, 2)),
                    cs=soft_coordinate,
                    soft_evidence=[evidence],
                    covmodel=self.covmodel,
                    covparam=self.covparam,
                    mode="vecchia",
                    max_parents=None,
                    ordering="input",
                    ep_damping=1.0,
                    ep_tolerance=1e-10,
                    ep_max_iterations=20,
                )
                result = operator.predict([])
                kernel = lambda value: (
                    norm.pdf(value, scale=np.sqrt(soft_variance))
                    * float(evidence_pdf(evidence, value)))
                points = list(evidence.abscissa) if isinstance(
                    evidence, TabulatedEvidence) else None
                normalization = quad(
                    kernel, -bound, bound, limit=500, points=points)[0]
                soft_mean = quad(
                    lambda value: value * kernel(value), -bound, bound,
                    limit=500, points=points)[0] / normalization
                soft_second = quad(
                    lambda value: value * value * kernel(value),
                    -bound, bound, limit=500, points=points)[0] / normalization
                expected_mean = regression * soft_mean
                expected_variance = (
                    conditional_variance
                    + regression ** 2 * (soft_second - soft_mean ** 2))
                np.testing.assert_allclose(
                    result.mean, [expected_mean], rtol=2e-7, atol=2e-7)
                np.testing.assert_allclose(
                    result.variance, [expected_variance], rtol=2e-7, atol=2e-7)
                self.assertTrue(result.metadata["ep_converged"])
                if isinstance(evidence, (LogNormalEvidence, TabulatedEvidence)):
                    tolerance = 1e-3
                elif isinstance(evidence, SkewNormalEvidence):
                    tolerance = 1e-5
                else:
                    tolerance = 1e-7
                self.assertLess(
                    result.metadata["maximum_relative_quadrature_error"], tolerance)

    def test_mixture_prediction_preserves_two_target_modes(self):
        evidence = GaussianMixtureEvidence(
            weights=(0.48, 0.52), means=(-1.8, 1.8), variances=(0.04, 0.04))
        operator = compile_vista_operator(
            np.array([[0.04, 0.0]]),
            ch=np.empty((0, 2)),
            cs=np.array([[0.0, 0.0]]),
            soft_evidence=[evidence],
            covmodel=["exponentialC"],
            covparam=[(1.0, [0.8])],
            mode="vecchia",
            max_parents=None,
            ordering="input",
            ep_damping=1.0,
            ep_tolerance=1e-11,
        )
        gaussian_ep = operator.predict([])
        mixture = operator.predict_mixture([])
        self.assertGreater(mixture.variance[0], gaussian_ep.variance[0])
        self.assertGreater(
            gaussian_ep.metadata["ep_relative_variance_residual"], 0.5)
        values = np.linspace(-3.0, 3.0, 1201)
        density = mixture.marginal_density(0, values)
        local_maxima = np.flatnonzero(
            (density[1:-1] > density[:-2]) & (density[1:-1] > density[2:])) + 1
        self.assertGreaterEqual(local_maxima.size, 2)
        self.assertEqual(mixture.metadata["retained_components"], 2)
        self.assertAlmostEqual(float(mixture.weights.sum()), 1.0, places=12)

    def test_ep_converged_factor_reuse_matches_legacy_refactor(self):
        arguments = dict(
            ck=self.ck,
            ch=self.ch,
            cs=self.cs,
            soft_evidence=[
                IntervalEvidence(2.0, 2.7),
                IntervalEvidence(1.6, 2.2),
                IntervalEvidence(3.2, 3.9),
            ],
            covmodel=self.covmodel,
            covparam=self.covparam,
            mode="vecchia",
            max_parents=4,
            ordering="random",
            random_state=17,
            ep_damping=0.7,
            ep_tolerance=1e-7,
            ep_max_iterations=50,
        )
        optimized = compile_vista_operator(
            **arguments, ep_reuse_converged_factor=True).predict(self.hard_values)
        legacy = compile_vista_operator(
            **arguments, ep_reuse_converged_factor=False).predict(self.hard_values)

        np.testing.assert_allclose(optimized.mean, legacy.mean, rtol=2e-7, atol=2e-7)
        np.testing.assert_allclose(
            optimized.variance, legacy.variance, rtol=2e-7, atol=2e-7)
        self.assertTrue(optimized.metadata["ep_reused_converged_factor"])
        self.assertEqual(
            optimized.metadata["factorizations"] + 1,
            legacy.metadata["factorizations"],
        )

    def test_streaming_rank_updates_match_batch_conditioning(self):
        prior_mean = np.array([0.2, -0.1, 0.4])
        prior_covariance = np.array([
            [1.4, 0.5, 0.2],
            [0.5, 1.1, 0.35],
            [0.2, 0.35, 0.9],
        ])
        nodes = np.array([0, 2])
        values = np.array([0.8, -0.2])
        noise = np.array([0.15, 0.08])
        state = StreamingRankState(prior_mean, prior_covariance)
        for node, value, variance in zip(nodes, values, noise):
            state.update(node, value, variance)

        observation_covariance = (
            prior_covariance[np.ix_(nodes, nodes)] + np.diag(noise))
        cross_covariance = prior_covariance[:, nodes]
        gain = np.linalg.solve(
            observation_covariance, cross_covariance.T).T
        expected_mean = prior_mean + gain @ (values - prior_mean[nodes])
        expected_covariance = prior_covariance - gain @ cross_covariance.T

        np.testing.assert_allclose(state.mean, expected_mean, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(
            state.covariance, expected_covariance, rtol=2e-12, atol=2e-12)
        self.assertEqual(state.updates, 2)

    def test_sparse_precision_streaming_matches_batch_conditioning(self):
        prior_mean = np.array([0.2, -0.1, 0.4])
        prior_covariance = np.array([
            [1.4, 0.5, 0.2],
            [0.5, 1.1, 0.35],
            [0.2, 0.35, 0.9],
        ])
        nodes = np.array([0, 2])
        values = np.array([0.8, -0.2])
        noise = np.array([0.15, 0.08])
        state = StreamingPrecisionState(
            np.linalg.inv(prior_covariance), prior_mean=prior_mean)
        diagnostics = [
            state.update(node, value, variance)
            for node, value, variance in zip(nodes, values, noise)
        ]

        observation_covariance = (
            prior_covariance[np.ix_(nodes, nodes)] + np.diag(noise))
        cross_covariance = prior_covariance[:, nodes]
        gain = np.linalg.solve(
            observation_covariance, cross_covariance.T).T
        expected_mean = prior_mean + gain @ (values - prior_mean[nodes])
        expected_covariance = prior_covariance - gain @ cross_covariance.T

        np.testing.assert_allclose(state.mean, expected_mean, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(
            state.posterior_variance(), np.diag(expected_covariance),
            rtol=2e-12, atol=2e-12)
        self.assertEqual(state.updates, 2)
        self.assertTrue(all(not item["graph_refactorized"] for item in diagnostics))

    def test_repeated_streaming_sites_equal_expanded_gaussian_likelihood(self):
        prior_precision = np.array([
            [2.0, -0.4, 0.0],
            [-0.4, 1.8, -0.2],
            [0.0, -0.2, 1.4],
        ])
        nodes = np.array([0, 1, 0, 2, 1, 0])
        values = np.array([1.0, -0.2, 0.8, 0.4, -0.1, 1.2])
        variances = np.array([0.2, 0.3, 0.1, 0.4, 0.25, 0.5])
        accumulator = StreamingSiteAccumulator(3)
        accumulator.update_many(nodes, values, variances)
        mean, _ = accumulator.posterior(prior_precision)

        observation_matrix = np.zeros((len(nodes), 3))
        observation_matrix[np.arange(len(nodes)), nodes] = 1.0
        inverse_noise = np.diag(1.0 / variances)
        expanded_precision = (
            prior_precision + observation_matrix.T @ inverse_noise @ observation_matrix)
        expanded_rhs = observation_matrix.T @ inverse_noise @ values
        expected = np.linalg.solve(expanded_precision, expanded_rhs)

        np.testing.assert_allclose(mean, expected, rtol=2e-12, atol=2e-12)
        self.assertEqual(accumulator.updates, len(nodes))
        np.testing.assert_array_equal(accumulator.count, [3, 2, 1])

    def test_calibration_report_contains_scores_and_coverage(self):
        report = gaussian_calibration_report(
            observed=[0.0, 1.0, -0.5, 0.25],
            mean=[0.1, 0.8, -0.4, 0.20],
            variance=[0.2, 0.3, 0.15, 0.1],
        )

        self.assertGreater(report["crps"], 0)
        self.assertGreater(report["nll"], 0)
        self.assertIn("coverage_0.9", report)
        self.assertIn("coverage_error_0.95", report)


if __name__ == "__main__":
    unittest.main()
