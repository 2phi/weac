"""Array-level tests for the shared layer-binning helper."""

from __future__ import annotations

import numpy as np
import pytest

from weac.parser.utils import bin_profile_to_layers, gradient_profile_to_layers


def _thicknesses(layers) -> list[float]:
    return [layer.h for layer in layers]


class TestNativeBinning:
    def test_half_mm_samples_merge_to_at_least_1mm(self):
        # 0.5 mm native step < 1 mm floor -> merge two samples per layer.
        distance = np.arange(0.0, 3.0, 0.5)  # 6 samples, cells all 0.5 mm
        density = np.full(distance.shape, 200.0)
        layers = bin_profile_to_layers(distance, density)

        assert len(layers) == 3
        assert all(layer.h >= 1.0 - 1e-9 for layer in layers)
        np.testing.assert_allclose(_thicknesses(layers), [1.0, 1.0, 1.0])

    def test_1p25mm_native_step_stays(self):
        distance = np.arange(0.0, 5.0, 1.25)  # cells 1.25 mm >= floor
        density = np.linspace(100.0, 200.0, distance.size)
        layers = bin_profile_to_layers(distance, density)

        # Each sample already clears the 1 mm floor -> one layer per sample.
        assert len(layers) == distance.size
        np.testing.assert_allclose(_thicknesses(layers), [1.25] * distance.size)

    def test_2p5mm_native_step_stays(self):
        distance = np.arange(0.0, 10.0, 2.5)
        density = np.full(distance.shape, 150.0)
        layers = bin_profile_to_layers(distance, density)

        assert len(layers) == distance.size
        np.testing.assert_allclose(_thicknesses(layers), [2.5] * distance.size)


class TestExplicitBinning:
    def test_explicit_10mm_bins(self):
        distance = np.arange(0.0, 20.0, 0.5)  # 40 samples of 0.5 mm = 20 mm
        density = np.full(distance.shape, 250.0)
        layers = bin_profile_to_layers(distance, density, layer_thickness_mm=10.0)

        assert len(layers) == 2
        np.testing.assert_allclose(_thicknesses(layers), [10.0, 10.0])

    def test_remainder_at_least_1mm_kept(self):
        # 45 * 0.5 = 22.5 mm -> 10, 10, remainder 2.5 mm (>= 1 mm) kept.
        distance = np.arange(0.0, 22.5, 0.5)
        density = np.full(distance.shape, 250.0)
        layers = bin_profile_to_layers(distance, density, layer_thickness_mm=10.0)

        assert len(layers) == 3
        np.testing.assert_allclose(_thicknesses(layers), [10.0, 10.0, 2.5])

    def test_remainder_below_1mm_merged(self):
        # 41 * 0.5 = 20.5 mm -> 10, 10, remainder 0.5 mm (< 1 mm) folded back.
        distance = np.arange(0.0, 20.5, 0.5)
        density = np.full(distance.shape, 250.0)
        layers = bin_profile_to_layers(distance, density, layer_thickness_mm=10.0)

        assert len(layers) == 2
        np.testing.assert_allclose(_thicknesses(layers), [10.0, 10.5])


class TestWeightedDensity:
    def test_thickness_weighted_mean_rho(self):
        # Non-uniform spacing so cell weights differ: cells = [1, 2, 2] mm
        # (last repeats the final spacing). One large bin groups all samples.
        distance = np.array([0.0, 1.0, 3.0])
        density = np.array([100.0, 200.0, 300.0])
        layers = bin_profile_to_layers(distance, density, layer_thickness_mm=100.0)

        assert len(layers) == 1
        expected = (1 * 100 + 2 * 200 + 2 * 300) / (1 + 2 + 2)  # 220.0
        assert layers[0].rho == pytest.approx(expected)

    def test_uniform_weights_reduce_to_mean(self):
        distance = np.arange(0.0, 3.0, 0.5)
        density = np.array([100.0, 140.0, 180.0, 220.0, 260.0, 300.0])
        layers = bin_profile_to_layers(distance, density)  # pairs per layer

        assert layers[0].rho == pytest.approx(np.mean(density[0:2]))
        assert layers[1].rho == pytest.approx(np.mean(density[2:4]))
        assert layers[2].rho == pytest.approx(np.mean(density[4:6]))


class TestThicknessConservation:
    @pytest.mark.parametrize("layer_thickness_mm", [None, 5.0, 10.0])
    def test_sum_of_h_conserved(self, layer_thickness_mm):
        distance = np.arange(0.0, 22.5, 0.5)
        density = np.linspace(120.0, 320.0, distance.size)
        # Cell model: each sample owns the spacing to the next; last repeats it.
        cell = np.diff(distance)
        total = float(cell.sum() + cell[-1])

        layers = bin_profile_to_layers(
            distance, density, layer_thickness_mm=layer_thickness_mm
        )
        assert sum(layer.h for layer in layers) == pytest.approx(total)

    def test_depth_scale_applied(self):
        distance = np.arange(0.0, 10.0, 0.5)
        density = np.full(distance.shape, 200.0)
        scale = np.cos(np.deg2rad(30.0))

        flat = bin_profile_to_layers(distance, density)
        scaled = bin_profile_to_layers(distance, density, depth_scale=scale)

        assert sum(layer.h for layer in scaled) == pytest.approx(
            sum(layer.h for layer in flat) * scale
        )


class TestGradientSegmentation:
    def test_sharp_jump_cuts_at_default_threshold(self):
        # 40 kg/m^3 step over a <=2.5 mm span -> |drho/dz| ~ 16 > T=8: a cut
        # must separate the 200 slab from the 240 slab.
        distance = np.arange(0.0, 20.0, 1.0)
        density = np.where(distance < 10.0, 200.0, 240.0)
        layers = gradient_profile_to_layers(distance, density)

        assert len(layers) >= 2
        assert layers[0].rho == pytest.approx(200.0)
        assert layers[-1].rho == pytest.approx(240.0)
        assert all(layer.h >= 1.0 - 1e-9 for layer in layers)

    def test_slow_ramp_stays_merged(self):
        # Same 40 kg/m^3 delta but spread over 300 mm -> ~0.13 kg/m^3/mm << T,
        # so every sample is below threshold and collapses into one layer.
        distance = np.arange(0.0, 300.0, 1.0)
        density = np.linspace(200.0, 240.0, distance.size)
        layers = gradient_profile_to_layers(distance, density)

        assert len(layers) == 1
        assert layers[0].rho == pytest.approx(float(np.mean(density)), abs=1.0)

    def test_steep_ramp_becomes_many_thin_layers(self):
        # Sustained 10 kg/m^3/mm slope > T -> every sample is a cut, floored to
        # a staircase of >= 1 mm layers.
        distance = np.arange(0.0, 30.0, 1.0)
        density = 200.0 + 10.0 * distance
        layers = gradient_profile_to_layers(distance, density)

        assert len(layers) == distance.size  # each 1 mm cell its own layer
        assert all(layer.h == pytest.approx(1.0) for layer in layers)

    def test_short_tail_uses_remaining_span(self):
        # Whole pack is 2 mm (< 2.5 mm span): the gradient must fall back to the
        # remaining span so the buried 260 layer is still cut out.
        distance = np.array([0.0, 0.5, 1.0, 1.5])
        density = np.array([200.0, 200.0, 260.0, 260.0])
        layers = gradient_profile_to_layers(distance, density)

        assert len(layers) == 2
        assert layers[0].rho == pytest.approx(200.0)
        assert layers[1].rho == pytest.approx(260.0)
        assert sum(layer.h for layer in layers) == pytest.approx(2.0)

    def test_threshold_override_suppresses_cut(self):
        # A jump that cuts at the default T merges when T is raised above it.
        distance = np.arange(0.0, 20.0, 1.0)
        density = np.where(distance < 10.0, 200.0, 240.0)
        layers = gradient_profile_to_layers(
            distance, density, threshold_kg_m3_per_mm=100.0
        )
        assert len(layers) == 1

    @pytest.mark.parametrize("threshold", [8.0, 20.0])
    def test_sum_of_h_conserved(self, threshold):
        distance = np.arange(0.0, 40.0, 0.5)
        density = np.linspace(150.0, 350.0, distance.size)
        cell = np.diff(distance)
        total = float(cell.sum() + cell[-1])

        layers = gradient_profile_to_layers(
            distance, density, threshold_kg_m3_per_mm=threshold
        )
        assert sum(layer.h for layer in layers) == pytest.approx(total)

    def test_depth_scale_applied(self):
        distance = np.arange(0.0, 30.0, 1.0)
        density = 200.0 + 10.0 * distance
        scale = np.cos(np.deg2rad(30.0))

        flat = gradient_profile_to_layers(distance, density)
        scaled = gradient_profile_to_layers(distance, density, depth_scale=scale)

        assert len(flat) == len(scaled)
        assert sum(layer.h for layer in scaled) == pytest.approx(
            sum(layer.h for layer in flat) * scale
        )
