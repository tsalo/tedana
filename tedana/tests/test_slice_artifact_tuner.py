"""Tests for tedana.workflows.slice_artifact_tuner."""

import numpy as np
import pandas as pd

from tedana.workflows.slice_artifact_tuner import (
    ANNOTATION_COLUMN,
    ANNOTATION_NUMERIC_COLUMN,
    _build_comparison_table,
    _make_default_component_labels,
    _summarize_alignment,
)


def test_make_default_component_labels():
    """Default labels align with component index."""
    labels = _make_default_component_labels(3)
    assert labels == ["ICA_00", "ICA_01", "ICA_02"]


def test_build_comparison_table_from_component_column():
    """Comparison table merges annotations by Component name."""
    component_labels = ["ICA_00", "ICA_01", "ICA_02"]
    metric_values = np.array([0.1, 0.5, 0.9])
    annotation_table = pd.DataFrame(
        {
            "Component": ["ICA_00", "ICA_01", "ICA_02"],
            ANNOTATION_COLUMN: ["1", "Unclear", "5"],
        }
    )
    comparison = _build_comparison_table(
        component_labels=component_labels,
        metric_values=metric_values,
        annotation_table=annotation_table,
    )

    assert comparison["Component"].tolist() == component_labels
    assert comparison[ANNOTATION_COLUMN].tolist() == ["1", "Unclear", "5"]
    assert comparison[ANNOTATION_NUMERIC_COLUMN].tolist()[0] == 1.0
    assert np.isnan(comparison[ANNOTATION_NUMERIC_COLUMN].tolist()[1])
    assert comparison[ANNOTATION_NUMERIC_COLUMN].tolist()[2] == 5.0


def test_summarize_alignment():
    """Summary computes correlation and counts on numeric annotations."""
    comparison = pd.DataFrame(
        {
            "Component": ["ICA_00", "ICA_01", "ICA_02", "ICA_03"],
            "slice_artifact_metric": [0.1, 0.2, 0.8, 0.9],
            ANNOTATION_COLUMN: ["1", "2", "4", "Unclear"],
            ANNOTATION_NUMERIC_COLUMN: [1.0, 2.0, 4.0, np.nan],
            "metric_rank_1to5": [1.0, 2.0, 4.0, 5.0],
        }
    )
    summary = _summarize_alignment(comparison)

    assert summary["n_total_components"] == 4
    assert summary["n_numeric_annotations"] == 3
    assert summary["n_unclear_annotations"] == 1
    assert summary["spearman_r"] is not None
    assert summary["pearson_r"] is not None
    assert "1" in summary["mean_metric_by_annotation"]
