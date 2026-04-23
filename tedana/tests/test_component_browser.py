"""Tests for tedana.workflows.component_browser."""

from pathlib import Path

import pandas as pd

from tedana.workflows.component_browser import (
    ANNOTATION_COLUMN,
    _build_annotation_export_table,
    _build_sorted_components,
    _collect_component_figures,
    _extract_component_index,
    _get_numeric_metric_columns,
    _get_run_prefix,
    _load_existing_annotations,
    _write_annotation_table,
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

    mapping = _collect_component_figures(figures_dir, run_prefix="")

    assert set(mapping.keys()) == {0}
    assert mapping[0].name == "comp_000.png"


def test_collect_component_figures_with_run_prefix(tmp_path: Path):
    """Only figures matching the run prefix are included."""
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    (figures_dir / "sub-01_comp_001.png").touch()
    (figures_dir / "sub-02_comp_001.png").touch()
    (figures_dir / "comp_001.png").touch()

    mapping = _collect_component_figures(figures_dir, run_prefix="sub-01")

    assert set(mapping.keys()) == {1}
    assert mapping[1].name == "sub-01_comp_001.png"


def test_get_run_prefix():
    """Run prefix is parsed from metrics TSV name."""
    prefixed = Path("/tmp/sub-01_desc-tedana_metrics.tsv")
    unprefixed = Path("/tmp/desc-tedana_metrics.tsv")
    nonstandard = Path("/tmp/component_metrics.tsv")
    assert _get_run_prefix(prefixed) == "sub-01"
    assert _get_run_prefix(unprefixed) == ""
    assert _get_run_prefix(nonstandard) == ""


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


def test_build_annotation_export_table():
    """Annotation export table includes labels and numeric helpers."""
    component_table = pd.DataFrame(
        {
            "Component": ["ICA_00", "ICA_01", "ICA_02"],
            "kappa": [5.0, 2.0, 9.0],
        }
    )
    figures_by_index = {0: Path("comp_000.png"), 1: Path("comp_001.png"), 2: Path("comp_002.png")}
    annotations = {"ICA_00": "1", "ICA_01": "Unclear"}

    table = _build_annotation_export_table(
        component_table=component_table,
        figures_by_index=figures_by_index,
        annotations=annotations,
    )

    assert ANNOTATION_COLUMN in table.columns
    assert table.loc[table["Component"] == "ICA_00", ANNOTATION_COLUMN].item() == "1"
    row_00 = table.loc[table["Component"] == "ICA_00"]
    row_01 = table.loc[table["Component"] == "ICA_01"]
    assert row_00["slice_artifact_annotation_numeric"].item() == 1.0
    assert row_01["slice_artifact_annotation_unclear"].item() == 1
    assert pd.isna(table.loc[table["Component"] == "ICA_02", ANNOTATION_COLUMN].item())


def test_load_existing_annotations(tmp_path: Path):
    """Existing annotations are loaded and filtered to supported labels."""
    annotation_table = pd.DataFrame(
        {
            "Component": ["ICA_00", "ICA_01", "ICA_02"],
            ANNOTATION_COLUMN: ["2", "Unclear", "bad-value"],
        }
    )
    annotation_path = tmp_path / "annotations.tsv"
    annotation_table.to_csv(annotation_path, sep="\t", index=False)

    annotations = _load_existing_annotations(annotation_path)
    assert annotations == {"ICA_00": "2", "ICA_01": "Unclear"}


def test_write_annotation_table_csv_and_tsv(tmp_path: Path):
    """Annotation writer supports csv and tsv output formats."""
    annotation_table = pd.DataFrame(
        {
            "Component": ["ICA_00"],
            ANNOTATION_COLUMN: ["5"],
        }
    )
    out_tsv = tmp_path / "annotations.tsv"
    out_csv = tmp_path / "annotations.csv"

    _write_annotation_table(annotation_table, out_tsv)
    _write_annotation_table(annotation_table, out_csv)

    assert out_tsv.exists()
    assert out_csv.exists()
    reloaded_tsv = pd.read_table(out_tsv)
    reloaded_csv = pd.read_csv(out_csv)
    assert reloaded_tsv[ANNOTATION_COLUMN].item() == "5"
    assert reloaded_csv[ANNOTATION_COLUMN].item() == "5"
