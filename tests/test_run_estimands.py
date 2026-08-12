"""Manufactured checks for independent-run Table 3 estimands."""

from __future__ import annotations

import unittest

import numpy as np

from src.public_data.run_estimands import (
    aggregate_independent_run_rates,
    gls_constant_heterogeneity,
    gls_constant_heterogeneity_interior,
    per_run_ratios,
    temporal_model_identifiability,
)


class RunEstimandTests(unittest.TestCase):
    def test_exposure_sum_and_covariance_are_exact_linear_maps(self):
        names = ("a", "reference")
        times = np.asarray([10.0, 30.0])
        rates = np.asarray([[2.0, 4.0], [6.0, 5.0]])
        covariance = np.asarray(
            [
                [0.04, 0.01, 0.008, 0.0],
                [0.01, 0.09, 0.0, 0.012],
                [0.008, 0.0, 0.16, 0.02],
                [0.0, 0.012, 0.02, 0.25],
            ]
        )
        result = aggregate_independent_run_rates(
            names, times, rates, covariance
        )
        np.testing.assert_allclose(result.summed_counts, [200.0, 190.0])
        np.testing.assert_allclose(
            result.aggregate_rates_counts_per_s, [5.0, 4.75]
        )
        expected = result.count_jacobian @ covariance @ result.count_jacobian.T
        np.testing.assert_allclose(result.summed_count_covariance, expected)
        np.testing.assert_allclose(
            result.aggregate_rate_covariance, expected / times.sum() ** 2
        )

    def test_aggregate_ratio_is_ratio_of_summed_counts_not_mean_run_ratio(self):
        names = ("a", "reference")
        rates = np.asarray([[1.0, 1.0], [9.0, 3.0]])
        times = np.asarray([90.0, 10.0])
        covariance = np.eye(4) * 0.01
        aggregate = aggregate_independent_run_rates(
            names, times, rates, covariance
        )
        aggregate_ratio = (
            aggregate.summed_counts[0] / aggregate.summed_counts[1]
        )
        self.assertAlmostEqual(aggregate_ratio, 1.5)
        self.assertNotAlmostEqual(aggregate_ratio, np.mean([1.0, 3.0]))

    def test_per_run_ratio_covariance_keeps_shared_nuisance_correlations(self):
        names = ("a", "reference")
        rates = np.asarray([[2.0, 4.0], [3.0, 5.0]])
        covariance = np.eye(4) * 0.04
        covariance[0, 2] = covariance[2, 0] = 0.015
        ratios = per_run_ratios(names, rates, covariance, "reference")
        self.assertEqual(ratios.values[1], 1.0)
        self.assertEqual(ratios.values[3], 1.0)
        self.assertEqual(ratios.covariance[1, 1], 0.0)
        self.assertNotEqual(ratios.covariance[0, 2], 0.0)

    def test_gls_detects_heterogeneous_run_rates(self):
        homogeneous = gls_constant_heterogeneity(
            [1.0, 1.02, 0.98], np.eye(3) * 0.1**2
        )
        heterogeneous = gls_constant_heterogeneity(
            [1.0, 1.8, 0.4], np.eye(3) * 0.1**2
        )
        self.assertGreater(homogeneous.p_value, 0.5)
        self.assertLess(heterogeneous.p_value, 1e-6)

    def test_boundary_aware_gls_matches_explicit_interior_subset(self):
        values = np.asarray([1.0, 1.2, 0.9, 20.0])
        covariance = np.asarray(
            [
                [0.04, 0.005, 0.002, 0.0],
                [0.005, 0.09, 0.003, 0.0],
                [0.002, 0.003, 0.16, 0.0],
                [0.0, 0.0, 0.0, 1.0e-20],
            ]
        )
        selected = gls_constant_heterogeneity_interior(
            values, covariance, [False, False, False, True]
        )
        explicit = gls_constant_heterogeneity(
            values[:3], covariance[:3, :3]
        )
        self.assertEqual(selected.status, "available_after_boundary_exclusion")
        self.assertEqual(selected.included_indices, (0, 1, 2))
        self.assertEqual(selected.excluded_boundary_indices, (3,))
        self.assertIsNotNone(selected.heterogeneity)
        self.assertEqual(selected.heterogeneity.degrees_of_freedom, 2)
        self.assertAlmostEqual(selected.heterogeneity.estimate, explicit.estimate)
        self.assertAlmostEqual(
            selected.heterogeneity.q_statistic, explicit.q_statistic
        )

    def test_boundary_aware_gls_fails_unavailable_below_two_interior_inputs(self):
        selected = gls_constant_heterogeneity_interior(
            [1.0, 2.0, 3.0], np.eye(3), [True, False, True]
        )
        self.assertEqual(
            selected.status,
            "unavailable_fewer_than_two_interior_inputs",
        )
        self.assertEqual(selected.included_indices, (1,))
        self.assertEqual(selected.excluded_boundary_indices, (0, 2))
        self.assertIsNone(selected.heterogeneity)

    def test_boundary_aware_gls_validates_mask_shape(self):
        with self.assertRaisesRegex(ValueError, "one value per estimand"):
            gls_constant_heterogeneity_interior(
                [1.0, 2.0], np.eye(2), [False]
            )

    def test_activation_and_power_models_fail_closed_without_history(self):
        rows = temporal_model_identifiability(
            [0.0, 86400.0, 172800.0, 259200.0],
            ["on", "on", "on", "on"],
        )
        by_name = {row["model"]: row for row in rows}
        self.assertTrue(by_name["constant_detected_rate"]["identifiable"])
        self.assertTrue(
            by_name["unconstrained_per_run_detected_rate"]["identifiable"]
        )
        self.assertFalse(
            by_name["reactor_power_correlated_scale"]["identifiable"]
        )
        self.assertFalse(
            by_name["activation_decay_known_half_life"]["identifiable"]
        )
        self.assertFalse(
            by_name["activation_decay_free_half_life"]["identifiable"]
        )

    def test_temporal_models_activate_only_with_declared_covariates(self):
        rows = temporal_model_identifiability(
            [0.0, 100.0, 250.0],
            ["on", "on", "off"],
            reactor_powers_mw=[85.0, 70.0, 0.0],
            source_history_available=True,
            known_half_life_s=120.0,
        )
        by_name = {row["model"]: row for row in rows}
        self.assertTrue(
            by_name["reactor_power_correlated_scale"]["identifiable"]
        )
        self.assertTrue(
            by_name["activation_decay_known_half_life"]["identifiable"]
        )
        self.assertTrue(
            by_name["activation_decay_free_half_life"]["identifiable"]
        )


if __name__ == "__main__":
    unittest.main()
