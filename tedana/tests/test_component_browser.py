"""Tests for tedana.workflows.component_browser."""

from pathlib import Path

import pandas as pd

from tedana.workflows.component_browser import (
    _build_sorted_components,
    _collect_component_figures,
    _extract_component_index,
    _get_numeric_metric_columns,
)


def test_extract_component_index():
    """Component index is parsed from trailing digits when available."""
    assert _extract_component_index("ICA_002", fallback=9) == 2
    assert _extract_component_index("PCA10", fallback=9) == 10
    assert _extract_component_index("component", fallback=9) == 9


def test_get_numeric_metric_columns():
    """Numeric metric detection ignores non-numeric columns."""
    component_table = pd.DataFrame(
        {
            "Component": ["ICA_00", "ICA_01"],
            "kappa": [10.0, 20.0],
            "classification": ["accepted", "rejected"],
            "mixed_numeric": ["1.0", "foo"],
        }
    )
    metrics = _get_numeric_metric_columns(component_table)
    assert "kappa" in metrics
    assert "mixed_numeric" in metrics
    assert "classification" not in metrics
    assert "Component" not in metrics


def test_collect_component_figures(tmp_path: Path):
    """Component figure files are indexed by component number."""
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    (figures_dir / "comp_000.png").touch()
    (figures_dir / "sub-01_comp_002.png").touch()
    (figures_dir / "ignore.svg").touch()

    mapping = _collect_component_figures(figures_dir)

    assert set(mapping.keys()) == {0, 2}
    assert mapping[0].name == "comp_000.png"
    assert mapping[2].name == "sub-01_comp_002.png"


def test_build_sorted_components():
    """Sorted component table is built with metric values and figure paths."""
    component_table = pd.DataFrame(
        {
            "Component": ["ICA_00", "ICA_01", "ICA_02"],
            "kappa": [5.0, 2.0, 9.0],
        }
    )
    figures_by_index = {0: Path("comp_000.png"), 1: Path("comp_001.png"), 2: Path("comp_002.png")}

    sorted_table = _build_sorted_components(
        component_table=component_table,
        metric="kappa",
        figures_by_index=figures_by_index,
        ascending=False,
    )

    assert sorted_table["Component"].tolist() == ["ICA_02", "ICA_00", "ICA_01"]
    assert sorted_table["metric_value"].tolist() == [9.0, 5.0, 2.0]
    assert sorted_table["figure_path"].tolist() == [
        Path("comp_002.png"),
        Path("comp_000.png"),
        Path("comp_001.png"),
    ]
