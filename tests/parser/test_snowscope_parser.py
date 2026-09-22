"""Integration tests for :class:`weac.parser.snowscope_parser.SnowScopeParser`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from weac.components import Layer
from weac.parser.snowscope_parser import SnowScopeParser


@pytest.fixture
def demo_data_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "demo" / "data"


@pytest.fixture
def snowscope_csv_path(demo_data_dir: Path) -> Path:
    path = demo_data_dir / "Profile671_SN00353.csv"
    assert path.is_file(), f"Demo SnowScope file missing: {path}"
    return path


class TestSnowScopeParser:
    def test_extract_profile_smoke(self, snowscope_csv_path: Path):
        parser = SnowScopeParser(str(snowscope_csv_path))
        profile = parser.extract_profile()

        assert profile.density_method == "HAGENMULLER2018"
        assert profile.depth_mm.shape == profile.density_kg_m3.shape
        assert profile.depth_mm.size == 690
        assert np.all(np.isfinite(profile.density_kg_m3))
        assert np.all(profile.density_kg_m3 >= 1.0)
        assert profile.depth_mm[0] == pytest.approx(1.0)
        assert profile.depth_mm[-1] == pytest.approx(690.0)

    def test_custom_semilog_coefficients(self, snowscope_csv_path: Path):
        parser = SnowScopeParser(str(snowscope_csv_path))
        default = parser.extract_profile()
        custom = parser.extract_profile(semilog_slope=10.0, semilog_intercept=5.0)

        assert custom.density_method == "CUSTOM"
        resistance = parser.penetration_resistance_kPa
        valid = np.isfinite(resistance) & (resistance > 0)
        expected = 10.0 * np.log(resistance[valid]) + 5.0
        np.testing.assert_allclose(custom.density_kg_m3[valid], expected)
        assert not np.allclose(default.density_kg_m3, custom.density_kg_m3)

    def test_extract_layers_bin_and_gradient(self, snowscope_csv_path: Path):
        parser = SnowScopeParser(str(snowscope_csv_path))
        layers_bin, methods_bin = parser.extract_layers(method="bin")
        layers_grad, methods_grad = parser.extract_layers(method="gradient")

        assert len(layers_bin) == 690
        assert len(layers_grad) < len(layers_bin)
        assert methods_bin == ["HAGENMULLER2018"] * len(layers_bin)
        assert methods_grad == ["HAGENMULLER2018"] * len(layers_grad)
        for layers in (layers_bin, layers_grad):
            assert all(isinstance(layer, Layer) for layer in layers)
            assert all(layer.h > 0 and layer.rho > 0 for layer in layers)
        assert sum(layer.h for layer in layers_bin) == pytest.approx(690.0)

    def test_extract_layers_unknown_method_raises(self, snowscope_csv_path: Path):
        parser = SnowScopeParser(str(snowscope_csv_path))
        with pytest.raises(ValueError, match='method must be "bin" or "gradient"'):
            parser.extract_layers(method="typo")  # type: ignore[arg-type]

    def test_semilog_pair_validation(self, snowscope_csv_path: Path):
        with pytest.raises(ValueError, match="semilog_slope"):
            SnowScopeParser(str(snowscope_csv_path), semilog_slope=1.0)
        with pytest.raises(ValueError, match="semilog_slope"):
            SnowScopeParser(str(snowscope_csv_path), semilog_intercept=2.0)

    def test_malformed_csv_missing_depth_table(self, tmp_path: Path):
        bad = tmp_path / "no_table.csv"
        bad.write_text("GENERAL INFO\nname,null\n", encoding="utf-8")
        with pytest.raises(ValueError, match="depth/hardness table"):
            SnowScopeParser(str(bad))

    def test_malformed_csv_no_numeric_rows(self, tmp_path: Path):
        bad = tmp_path / "empty_table.csv"
        bad.write_text(
            "SCOPE PROFILE\ndepth (mm),hardness (kPa),\nnot_a_number,also_bad,null\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="No numeric depth/hardness"):
            SnowScopeParser(str(bad))

    def test_non_positive_hardness_raises(self, tmp_path: Path):
        bad = tmp_path / "zero_hardness.csv"
        bad.write_text(
            "SCOPE PROFILE\n"
            "depth (mm),hardness (kPa),\n"
            "1.0,10.0,null\n"
            "2.0,0.0,null\n"
            "3.0,5.0,null\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="non-positive hardness"):
            SnowScopeParser(str(bad))
