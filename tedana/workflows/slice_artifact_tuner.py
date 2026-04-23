"""Utilities to iterate on slice artifact metrics against manual annotations."""

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nb
import numpy as np
import pandas as pd
from scipy import stats

ANNOTATION_COLUMN = "slice_artifact_annotation"
ANNOTATION_NUMERIC_COLUMN = "slice_artifact_annotation_numeric"


def _get_parser():
    """Build parser for slice artifact metric tuning."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Recompute a slice artifact metric from ICA component maps and compare "
            "the results against manual annotations."
        ),
    )
    parser.add_argument(
        "--component-maps",
        required=True,
        help="Path to a 4D ICA component map file (e.g., desc-ICA_components.nii.gz).",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help=(
            "Path to exported annotations from tedana_component_browser "
            "(TSV/CSV with component labels)."
        ),
    )
    parser.add_argument(
        "--metrics-tsv",
        default=None,
        help="Optional metrics TSV to get component labels from the 'Component' column.",
    )
    parser.add_argument(
        "--mask",
        default=None,
        help=(
            "Optional brain mask image. If omitted, a mask is derived from nonzero "
            "voxels in component maps."
        ),
    )
    parser.add_argument(
        "--slice-axis",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Optional slice axis override. If omitted, inferred from image header.",
    )
    parser.add_argument(
        "--metric-function",
        default="tedana.metrics.spatial:calculate_slice_artifact",
        help=(
            "Function used to compute the metric. Format module:function. "
            "Expected signature: (betas, mask_img, slice_axis) -> ndarray."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for comparison table and summary JSON.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Prefix prepended to output files.",
    )
    return parser


def _read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or TSV table based on file extension."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_table(path)


def _load_metric_function(function_spec: str):
    """Load metric function from module:function specification."""
    if ":" not in function_spec:
        raise ValueError(
            f"metric-function must be in module:function format. Received '{function_spec}'"
        )
    module_name, function_name = function_spec.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if function is None or not callable(function):
        raise ValueError(f"Could not load callable '{function_name}' from module '{module_name}'.")
    return function


def _determine_slice_axis(component_img: nb.Nifti1Image, override: int = None) -> int:
    """Determine slice axis from override or NIfTI header."""
    if override is not None:
        return override
    slice_axis = component_img.header.get_dim_info()[2]
    if slice_axis is None:
        slice_axis = 2
    return slice_axis


def _derive_mask(component_data: np.ndarray) -> np.ndarray:
    """Derive a binary mask from nonzero component-map voxels."""
    mask = np.any(np.abs(component_data) > 0, axis=3)
    if not np.any(mask):
        raise ValueError("Derived mask is empty. Provide --mask explicitly.")
    return mask


def _make_default_component_labels(n_components: int) -> List[str]:
    """Create default ICA labels aligned to component indices."""
    width = max(2, len(str(max(n_components - 1, 0))))
    return [f"ICA_{idx:0{width}d}" for idx in range(n_components)]


def _load_component_labels(
    *,
    n_components: int,
    metrics_tsv_path: Path = None,
) -> List[str]:
    """Load component labels from metrics table or create defaults."""
    if metrics_tsv_path is None:
        return _make_default_component_labels(n_components)

    metrics_table = _read_table(metrics_tsv_path)
    if "Component" not in metrics_table.columns:
        raise ValueError(f"'Component' column not found in metrics table: {metrics_tsv_path}")
    if len(metrics_table) != n_components:
        raise ValueError(
            "Number of rows in metrics TSV ({}) does not match number of components ({})".format(
                len(metrics_table), n_components
            )
        )
    return metrics_table["Component"].astype(str).tolist()


def _coerce_annotation_numeric(annotation_table: pd.DataFrame) -> pd.Series:
    """Extract numeric annotation labels (1-5), leaving 'Unclear' as NaN."""
    if ANNOTATION_NUMERIC_COLUMN in annotation_table.columns:
        return pd.to_numeric(annotation_table[ANNOTATION_NUMERIC_COLUMN], errors="coerce")
    if ANNOTATION_COLUMN in annotation_table.columns:
        return pd.to_numeric(annotation_table[ANNOTATION_COLUMN], errors="coerce")
    return pd.Series(np.nan, index=annotation_table.index)


def _build_comparison_table(
    *,
    component_labels: List[str],
    metric_values: np.ndarray,
    annotation_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build a component-level table with metric values and manual annotations."""
    comparison = pd.DataFrame(
        {
            "Component": component_labels,
            "component_index": np.arange(len(component_labels), dtype=int),
            "slice_artifact_metric": metric_values.astype(float),
        }
    )

    annotation_table = annotation_table.copy()
    if "Component" not in annotation_table.columns:
        if "component_index" in annotation_table.columns:
            annotation_table["component_index"] = pd.to_numeric(
                annotation_table["component_index"], errors="coerce"
            )
            merged = comparison.merge(
                annotation_table,
                on="component_index",
                how="left",
                suffixes=("", "_ann"),
            )
        else:
            raise ValueError(
                "Annotation file must include either 'Component' or 'component_index' columns."
            )
    else:
        annotation_table["Component"] = annotation_table["Component"].astype(str)
        merged = comparison.merge(
            annotation_table,
            on="Component",
            how="left",
            suffixes=("", "_ann"),
        )

    if "component_index_ann" in merged.columns:
        merged = merged.drop(columns=["component_index_ann"])
    if ANNOTATION_COLUMN not in merged.columns:
        merged[ANNOTATION_COLUMN] = np.nan

    merged[ANNOTATION_NUMERIC_COLUMN] = _coerce_annotation_numeric(merged)
    merged["metric_rank"] = merged["slice_artifact_metric"].rank(method="average", ascending=True)
    if len(merged) > 1:
        merged["metric_rank_1to5"] = 1 + 4 * (merged["metric_rank"] - 1) / (len(merged) - 1)
    else:
        merged["metric_rank_1to5"] = 3.0
    merged["annotation_minus_rank_metric"] = (
        merged[ANNOTATION_NUMERIC_COLUMN] - merged["metric_rank_1to5"]
    )
    return merged


def _summarize_alignment(comparison_table: pd.DataFrame) -> Dict:
    """Summarize agreement between metric values and manual labels."""
    summary = {
        "n_total_components": int(len(comparison_table)),
        "n_unclear_annotations": int((comparison_table[ANNOTATION_COLUMN] == "Unclear").sum()),
    }
    numeric = comparison_table[ANNOTATION_NUMERIC_COLUMN].notna()
    summary["n_numeric_annotations"] = int(numeric.sum())

    if summary["n_numeric_annotations"] < 2:
        summary["spearman_r"] = None
        summary["spearman_p"] = None
        summary["pearson_r"] = None
        summary["pearson_p"] = None
    else:
        spearman_r, spearman_p = stats.spearmanr(
            comparison_table.loc[numeric, "slice_artifact_metric"],
            comparison_table.loc[numeric, ANNOTATION_NUMERIC_COLUMN],
        )
        pearson_r, pearson_p = stats.pearsonr(
            comparison_table.loc[numeric, "metric_rank_1to5"],
            comparison_table.loc[numeric, ANNOTATION_NUMERIC_COLUMN],
        )
        summary["spearman_r"] = float(spearman_r)
        summary["spearman_p"] = float(spearman_p)
        summary["pearson_r"] = float(pearson_r)
        summary["pearson_p"] = float(pearson_p)

    means = (
        comparison_table.groupby(ANNOTATION_COLUMN)["slice_artifact_metric"]
        .mean()
        .dropna()
        .to_dict()
    )
    summary["mean_metric_by_annotation"] = {str(k): float(v) for k, v in means.items()}
    return summary


def _default_output_paths(
    *,
    out_dir: Path,
    prefix: str,
) -> Tuple[Path, Path]:
    """Construct default output paths."""
    comp_path = out_dir / f"{prefix}slice_artifact_metric_comparison.tsv"
    summary_path = out_dir / f"{prefix}slice_artifact_metric_summary.json"
    return comp_path, summary_path


def tune_slice_artifact_workflow(
    *,
    component_maps: str,
    annotations: str,
    metrics_tsv: str = None,
    mask: str = None,
    slice_axis: int = None,
    metric_function: str = "tedana.metrics.spatial:calculate_slice_artifact",
    out_dir: str = None,
    prefix: str = "",
) -> Tuple[pd.DataFrame, Dict]:
    """Recompute slice artifact metric and compare to manual annotations."""
    component_maps_path = Path(component_maps).expanduser().resolve()
    annotations_path = Path(annotations).expanduser().resolve()
    metrics_tsv_path = Path(metrics_tsv).expanduser().resolve() if metrics_tsv else None
    mask_path = Path(mask).expanduser().resolve() if mask else None

    if not component_maps_path.exists():
        raise ValueError(f"Component map file does not exist: {component_maps_path}")
    if not annotations_path.exists():
        raise ValueError(f"Annotation file does not exist: {annotations_path}")
    if metrics_tsv_path is not None and not metrics_tsv_path.exists():
        raise ValueError(f"Metrics TSV file does not exist: {metrics_tsv_path}")
    if mask_path is not None and not mask_path.exists():
        raise ValueError(f"Mask file does not exist: {mask_path}")

    component_img = nb.load(component_maps_path)
    component_data = np.asanyarray(component_img.dataobj)
    if component_data.ndim != 4:
        raise ValueError(
            f"Component map image must be 4D (X, Y, Z, C). Received shape {component_data.shape}"
        )
    n_components = component_data.shape[3]

    if mask_path is None:
        mask_arr = _derive_mask(component_data)
        mask_img = nb.Nifti1Image(mask_arr.astype(np.uint8), component_img.affine)
    else:
        input_mask_img = nb.load(mask_path)
        mask_arr = np.asanyarray(input_mask_img.dataobj).astype(bool)
        if mask_arr.shape != component_data.shape[:3]:
            raise ValueError(
                "Mask shape {} does not match component map shape {}".format(
                    mask_arr.shape, component_data.shape[:3]
                )
            )
        # Force a binary mask image for nilearn masking utilities.
        mask_img = nb.Nifti1Image(mask_arr.astype(np.uint8), input_mask_img.affine)

    metric_callable = _load_metric_function(metric_function)
    axis = _determine_slice_axis(component_img, override=slice_axis)
    betas = component_data[mask_arr, :]
    metric_values = metric_callable(betas=betas, mask_img=mask_img, slice_axis=axis)
    metric_values = np.asarray(metric_values)
    if metric_values.shape != (n_components,):
        raise ValueError(
            "Metric function must return one value per component. "
            f"Expected {(n_components,)}, got {metric_values.shape}"
        )

    component_labels = _load_component_labels(
        n_components=n_components,
        metrics_tsv_path=metrics_tsv_path,
    )
    annotation_table = _read_table(annotations_path)
    comparison_table = _build_comparison_table(
        component_labels=component_labels,
        metric_values=metric_values,
        annotation_table=annotation_table,
    )
    summary = _summarize_alignment(comparison_table)

    output_dir = Path(out_dir).expanduser().resolve() if out_dir else annotations_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path, summary_path = _default_output_paths(out_dir=output_dir, prefix=prefix)
    comparison_table.to_csv(comparison_path, sep="\t", index=False)
    with summary_path.open("w", encoding="utf-8") as fo:
        json.dump(summary, fo, indent=2)

    print(f"Saved comparison table: {comparison_path}")
    print(f"Saved summary JSON: {summary_path}")
    print(
        "Numeric annotations: {}/{} | Spearman r: {}".format(
            summary["n_numeric_annotations"],
            summary["n_total_components"],
            summary["spearman_r"],
        )
    )
    return comparison_table, summary


def _main(argv=None):
    """Entry point for command line use."""
    parser = _get_parser()
    args = parser.parse_args(argv)
    tune_slice_artifact_workflow(
        component_maps=args.component_maps,
        annotations=args.annotations,
        metrics_tsv=args.metrics_tsv,
        mask=args.mask,
        slice_axis=args.slice_axis,
        metric_function=args.metric_function,
        out_dir=args.out_dir,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    _main()
