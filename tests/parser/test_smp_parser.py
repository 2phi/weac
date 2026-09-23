"""Integration tests for :class:`weac.parser.smp_parser.SMPParser`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weac.components import Layer
from weac.parser.smp_parser import SMPParser
from weac.parser.utils import detect_knappe_surface


@pytest.fixture
def demo_data_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "demo" / "data"


@pytest.fixture
def smp_pnt_path(demo_data_dir: Path) -> Path:
    path = demo_data_dir / "S36N2470.PNT"
    assert path.is_file(), f"Demo SMP file missing: {path}"
    return path


class TestSMPParser:
    def test_extract_profile_smoke(self, smp_pnt_path: Path):
        parser = SMPParser(str(smp_pnt_path))
        profile = parser.extract_profile()

        assert profile.density_method == "P2015"
        assert profile.depth_mm.shape == profile.density_kg_m3.shape
        assert profile.depth_mm.size > 0
        assert np.all(np.isfinite(profile.density_kg_m3))
        assert np.all(profile.density_kg_m3 > 0)
        assert np.all(np.isfinite(profile.penetration_resistance_kPa))
        assert np.all(np.isfinite(profile.ssa_m2_kg))

    def test_extract_profile_density_method_override(self, smp_pnt_path: Path):
        parser = SMPParser(str(smp_pnt_path), density_method="P2015")
        p2015 = parser.extract_profile()
        cr2020 = parser.extract_profile(density_method="CR2020")

        assert cr2020.density_method == "CR2020"
        # Parameterizations use different Loewe windows, so sample counts differ.
        assert p2015.density_kg_m3.shape != cr2020.density_kg_m3.shape
        assert float(np.mean(p2015.density_kg_m3)) != pytest.approx(
            float(np.mean(cr2020.density_kg_m3))
        )

    def test_extract_layers_bin_and_gradient(self, smp_pnt_path: Path):
        parser = SMPParser(str(smp_pnt_path))
        layers_bin, methods_bin = parser.extract_layers(method="bin")
        layers_grad, methods_grad = parser.extract_layers(method="gradient")

        assert len(layers_bin) > len(layers_grad)
        assert methods_bin == ["P2015"] * len(layers_bin)
        assert methods_grad == ["P2015"] * len(layers_grad)
        for layers in (layers_bin, layers_grad):
            assert all(isinstance(layer, Layer) for layer in layers)
            assert all(layer.h > 0 and layer.rho > 0 for layer in layers)
            assert sum(layer.h for layer in layers) == pytest.approx(
                float(profile_depth_sum(parser)), rel=1e-4
            )

    def test_extract_layers_unknown_method_raises(self, smp_pnt_path: Path):
        parser = SMPParser(str(smp_pnt_path))
        with pytest.raises(ValueError, match='method must be "bin" or "gradient"'):
            parser.extract_layers(method="typo")  # type: ignore[arg-type]

    def test_extract_layers_non_positive_density_raises(self, smp_pnt_path: Path):
        parser = SMPParser(str(smp_pnt_path))
        profile = parser.extract_profile(density_method="P2015")
        bad = profile.density_kg_m3.copy()
        bad[5] = -12.0

        original_extract_profile = parser.extract_profile

        def fake_extract_profile(*, density_method=None):
            method = (
                density_method if density_method is not None else parser.density_method
            )
            out = original_extract_profile(density_method=method)
            if method == "P2015":
                return out.__class__(
                    coordinates=out.coordinates,
                    depth_mm=out.depth_mm,
                    penetration_resistance_kPa=out.penetration_resistance_kPa,
                    density_kg_m3=bad,
                    ssa_m2_kg=out.ssa_m2_kg,
                    density_method=out.density_method,
                )
            return out

        parser.extract_profile = fake_extract_profile  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="not strictly positive") as exc:
            parser.extract_layers(density_method="P2015", method="bin")
        assert "P2015" in str(exc.value)
        assert "kg/m³" in str(exc.value)

    def test_surface_mm_out_of_range_raises(self, smp_pnt_path: Path):
        parser = SMPParser(str(smp_pnt_path))
        length = parser.loaded_profile.recording_length
        with pytest.raises(ValueError, match="surface_mm"):
            SMPParser(str(smp_pnt_path), surface_mm=length + 1.0)

    def test_knappe_surface_crops_deeper_than_snowmicropyn(self, smp_pnt_path: Path):
        default = SMPParser(str(smp_pnt_path), density_method="CR2020")
        knappe = SMPParser(
            str(smp_pnt_path), density_method="CR2020", surface_method="knappe"
        )

        assert default.surface_method == "snowmicropyn"
        assert knappe.surface_method == "knappe"
        # Knappe surface on this pit sits below snowmicropyn's surface.
        assert knappe.loaded_profile.surface == pytest.approx(132.492, abs=0.01)
        assert knappe.loaded_profile.surface > default.loaded_profile.surface

        prof_default = default.extract_profile()
        prof_knappe = knappe.extract_profile()
        assert prof_knappe.density_method == "CR2020"
        assert prof_knappe.depth_mm.size > 0
        assert np.all(np.isfinite(prof_knappe.penetration_resistance_kPa))
        assert np.all(prof_knappe.density_kg_m3 > 0)
        assert prof_knappe.depth_mm[-1] < prof_default.depth_mm[-1]

    def test_surface_mm_overrides_surface_method(self, smp_pnt_path: Path):
        parser = SMPParser(str(smp_pnt_path), surface_mm=50.0, surface_method="knappe")
        assert parser.surface_method == "manual"
        assert parser.loaded_profile.surface == pytest.approx(50.0)

    def test_manual_surface_method_sets_surface_mm(self, smp_pnt_path: Path):
        parser = SMPParser(str(smp_pnt_path), surface_method="manual", surface_mm=50.0)
        assert parser.surface_method == "manual"
        assert parser.loaded_profile.surface == pytest.approx(50.0)

    def test_manual_surface_method_requires_surface_mm(self, smp_pnt_path: Path):
        with pytest.raises(ValueError, match="surface_mm is required"):
            SMPParser(str(smp_pnt_path), surface_method="manual")

    def test_unknown_surface_method_raises(self, smp_pnt_path: Path):
        with pytest.raises(ValueError, match="surface_method"):
            SMPParser(str(smp_pnt_path), surface_method="typo")  # type: ignore[arg-type]

    def test_knappe_surface_falls_back_when_profile_is_shorter_than_window(self):
        samples = {
            "distance": np.array([1.5, 2.0]),
            "force": np.array([0.0, 0.2]),
        }
        assert detect_knappe_surface(samples) == pytest.approx(1.5)


def profile_depth_sum(parser: SMPParser) -> float:
    """Total slope-normal thickness implied by the windowed SMP depth axis."""
    profile = parser.extract_profile()
    depth = profile.depth_mm
    cell = np.diff(depth)
    return float(cell.sum() + cell[-1])
