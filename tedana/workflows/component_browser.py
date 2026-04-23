"""Interactive browser for tedana component figures sorted by metric values."""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

LGR = logging.getLogger("GENERAL")
ANNOTATION_OPTIONS = ["1", "2", "3", "4", "5", "Unclear"]
ANNOTATION_COLUMN = "slice_artifact_annotation"


def _get_parser():
    """Build parser for the component browser command."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Browse tedana component static figures sorted by a selected metric "
            "from a metrics TSV file."
        ),
    )
    parser.add_argument(
        "--metrics-tsv",
        required=True,
        help="Path to tedana metrics TSV (e.g., desc-tedana_metrics.tsv).",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help=(
            "Path to the figures directory containing comp_###.png files. "
            "Defaults to a sibling 'figures' directory next to --metrics-tsv."
        ),
    )
    parser.add_argument(
        "--metric",
        default=None,
        help=(
            "Initial metric to sort by. If omitted, the first numeric metric in "
            "the TSV is used."
        ),
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort selected metric in descending order.",
    )
    parser.add_argument(
        "--annotations-out",
        default=None,
        help=(
            "Output file for saved annotations. Defaults to "
            "'slice_artifact_annotations.tsv' next to --metrics-tsv."
        ),
    )
    parser.add_argument(
        "--annotations-in",
        default=None,
        help=(
            "Optional existing annotation file to preload labels from. "
            "If omitted and --annotations-out exists, that file is used."
        ),
    )
    return parser


def _extract_component_index(component_name: str, fallback: int) -> int:
    """Extract numeric index from component name (e.g., ICA_12 -> 12)."""
    match = re.search(r"(\d+)$", str(component_name))
    if match:
        return int(match.group(1))
    return int(fallback)


def _get_numeric_metric_columns(component_table: pd.DataFrame) -> List[str]:
    """Return column names that contain at least one numeric value."""
    numeric_cols = []
    for col in component_table.columns:
        if col == "Component":
            continue
        values = pd.to_numeric(component_table[col], errors="coerce")
        if values.notna().any():
            numeric_cols.append(col)
    return numeric_cols


def _get_run_prefix(metrics_path: Path) -> str:
    """Extract run prefix from metrics file stem before '_desc-tedana'."""
    stem = metrics_path.stem
    token = "_desc-tedana"
    if token in stem:
        return stem.split(token, 1)[0]
    return ""


def _collect_component_figures(figures_dir: Path, run_prefix: str) -> Dict[int, Path]:
    """Map component index to static figure path."""
    if run_prefix:
        figure_pattern = re.compile(rf"^{re.escape(run_prefix)}_comp_(\d+)\.png$")
    else:
        figure_pattern = re.compile(r"^comp_(\d+)\.png$")

    figures_by_index: Dict[int, Path] = {}
    for file_ in sorted(figures_dir.glob("*comp_*.png")):
        match = figure_pattern.match(file_.name)
        if not match:
            continue
        comp_idx = int(match.group(1))
        figures_by_index.setdefault(comp_idx, file_)
    return figures_by_index


def _component_index_table(component_table: pd.DataFrame) -> pd.DataFrame:
    """Return a component table with extracted component indices."""
    component_index_table = component_table.copy()
    component_index_table["component_index"] = [
        _extract_component_index(name, fallback=idx)
        for idx, name in enumerate(component_index_table["Component"].tolist())
    ]
    return component_index_table


def _read_table(path: Path) -> pd.DataFrame:
    """Read CSV/TSV annotation table based on extension."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_table(path)


def _load_existing_annotations(path: Path) -> Dict[str, str]:
    """Load annotation labels by Component from a file."""
    if not path.exists():
        return {}

    annotation_table = _read_table(path)
    if (
        "Component" not in annotation_table.columns
        or ANNOTATION_COLUMN not in annotation_table.columns
    ):
        raise ValueError(
            f"Annotation file must include 'Component' and '{ANNOTATION_COLUMN}' columns: {path}"
        )

    annotations = {}
    for _, row in annotation_table.iterrows():
        component = row["Component"]
        annotation = row[ANNOTATION_COLUMN]
        if pd.isna(annotation):
            continue
        annotation = str(annotation).strip()
        if annotation not in ANNOTATION_OPTIONS:
            continue
        annotations[str(component)] = annotation
    return annotations


def _build_annotation_export_table(
    *,
    component_table: pd.DataFrame,
    figures_by_index: Dict[int, Path],
    annotations: Dict[str, str],
) -> pd.DataFrame:
    """Build export table for manual slice artifact annotations."""
    export_table = _component_index_table(component_table)
    export_table["figure_path"] = export_table["component_index"].map(figures_by_index)
    export_table[ANNOTATION_COLUMN] = export_table["Component"].map(annotations)
    export_table["slice_artifact_annotation_numeric"] = pd.to_numeric(
        export_table[ANNOTATION_COLUMN], errors="coerce"
    )
    export_table["slice_artifact_annotation_unclear"] = (
        export_table[ANNOTATION_COLUMN] == "Unclear"
    ).astype(int)
    return export_table[
        [
            "Component",
            "component_index",
            "figure_path",
            ANNOTATION_COLUMN,
            "slice_artifact_annotation_numeric",
            "slice_artifact_annotation_unclear",
        ]
    ]


def _write_annotation_table(annotation_table: pd.DataFrame, output_path: Path):
    """Write annotation table to CSV/TSV based on extension."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        annotation_table.to_csv(output_path, index=False)
    else:
        annotation_table.to_csv(output_path, sep="\t", index=False)


def _build_sorted_components(
    *,
    component_table: pd.DataFrame,
    metric: str,
    figures_by_index: Dict[int, Path],
    ascending: bool,
) -> pd.DataFrame:
    """Create a sorted table with figure paths for available components."""
    if metric not in component_table.columns:
        raise ValueError(f"Metric '{metric}' not found in component table.")

    sortable = component_table.copy()
    sortable["component_index"] = [
        _extract_component_index(name, fallback=idx)
        for idx, name in enumerate(sortable["Component"].tolist())
    ]
    sortable["metric_value"] = pd.to_numeric(sortable[metric], errors="coerce")
    sortable["figure_path"] = sortable["component_index"].map(figures_by_index)
    sortable = sortable[sortable["figure_path"].notna()].copy()
    sortable = sortable.sort_values("metric_value", ascending=ascending, na_position="last")
    sortable = sortable.reset_index(drop=True)
    return sortable


class _ComponentBrowserApp:
    """Simple Tk app for browsing tedana component figures."""

    def __init__(
        self,
        root,
        *,
        component_table: pd.DataFrame,
        metrics: List[str],
        figures_by_index: Dict[int, Path],
        initial_metric: str,
        descending: bool,
        annotations_out_path: Path,
        existing_annotations: Dict[str, str],
    ):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.component_table = component_table
        self.metrics = metrics
        self.figures_by_index = figures_by_index
        self.current_metric = initial_metric
        self.descending = descending
        self.current_idx = 0
        self.sorted_components = pd.DataFrame()
        self._photo = None
        self.annotations_out_path = annotations_out_path
        self.annotations = dict(existing_annotations)
        self._annotation_update_in_progress = False

        root.title("tedana Component Figure Browser")
        root.geometry("1200x900")

        control_frame = ttk.Frame(root, padding=8)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="Metric:").pack(side=tk.LEFT, padx=(0, 6))
        self.metric_var = tk.StringVar(value=self.current_metric)
        self.metric_box = ttk.Combobox(
            control_frame,
            textvariable=self.metric_var,
            values=self.metrics,
            state="readonly",
            width=40,
        )
        self.metric_box.pack(side=tk.LEFT, padx=(0, 12))
        self.metric_box.bind("<<ComboboxSelected>>", self._on_metric_changed)

        self.desc_var = tk.BooleanVar(value=self.descending)
        self.desc_check = ttk.Checkbutton(
            control_frame,
            text="Descending",
            variable=self.desc_var,
            command=self._on_sort_order_changed,
        )
        self.desc_check.pack(side=tk.LEFT, padx=(0, 12))

        self.status_var = tk.StringVar()
        ttk.Label(control_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=(0, 12))

        nav_frame = ttk.Frame(root, padding=(8, 0, 8, 8))
        nav_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(nav_frame, text="Previous", command=lambda: self._move(-1)).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(nav_frame, text="Next", command=lambda: self._move(1)).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        ttk.Label(
            nav_frame,
            text="Use mouse wheel or arrow keys to scroll components.",
        ).pack(side=tk.LEFT)

        annotation_frame = ttk.Frame(root, padding=(8, 0, 8, 8))
        annotation_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(
            annotation_frame,
            text="Slice artifact annotation (1=no artifact, 5=clear artifact):",
        ).pack(side=tk.LEFT, padx=(0, 8))
        self.annotation_var = tk.StringVar(value="")
        self.annotation_var.trace_add("write", self._on_annotation_changed)
        for option in ANNOTATION_OPTIONS:
            ttk.Radiobutton(
                annotation_frame,
                text=option,
                value=option,
                variable=self.annotation_var,
            ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(annotation_frame, text="Clear", command=self._clear_annotation).pack(
            side=tk.LEFT, padx=(8, 8)
        )
        ttk.Button(
            annotation_frame,
            text="Export annotations",
            command=self._export_annotations,
        ).pack(side=tk.LEFT)

        self.export_status_var = tk.StringVar(value=f"Output: {self.annotations_out_path}")
        ttk.Label(root, textvariable=self.export_status_var).pack(
            side=tk.TOP, anchor="w", padx=8, pady=(0, 4)
        )

        self.image_label = ttk.Label(root, anchor="center")
        self.image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.root.bind("<Left>", lambda _event: self._move(-1))
        self.root.bind("<Right>", lambda _event: self._move(1))
        self.root.bind("<Up>", lambda _event: self._move(-1))
        self.root.bind("<Down>", lambda _event: self._move(1))
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        self.root.bind("<Button-4>", lambda _event: self._move(-1))
        self.root.bind("<Button-5>", lambda _event: self._move(1))
        self.root.bind("1", lambda _event: self._set_annotation("1"))
        self.root.bind("2", lambda _event: self._set_annotation("2"))
        self.root.bind("3", lambda _event: self._set_annotation("3"))
        self.root.bind("4", lambda _event: self._set_annotation("4"))
        self.root.bind("5", lambda _event: self._set_annotation("5"))
        self.root.bind("u", lambda _event: self._set_annotation("Unclear"))
        self.root.bind("U", lambda _event: self._set_annotation("Unclear"))
        self.root.bind("<Control-s>", lambda _event: self._export_annotations())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._refresh_sorted_components(reset_index=True)

    def _on_metric_changed(self, _event):
        self.current_metric = self.metric_var.get()
        self._refresh_sorted_components(reset_index=True)

    def _on_sort_order_changed(self):
        self.descending = self.desc_var.get()
        self._refresh_sorted_components(reset_index=True)

    def _on_mousewheel(self, event):
        if event.delta > 0:
            self._move(-1)
        elif event.delta < 0:
            self._move(1)

    def _current_component_name(self):
        if self.sorted_components.empty:
            return None
        return str(self.sorted_components.iloc[self.current_idx]["Component"])

    def _on_annotation_changed(self, *_args):
        if self._annotation_update_in_progress:
            return
        component_name = self._current_component_name()
        if component_name is None:
            return
        value = self.annotation_var.get()
        if value in ANNOTATION_OPTIONS:
            self.annotations[component_name] = value
        else:
            self.annotations.pop(component_name, None)
        self._display_current()

    def _set_annotation(self, value: str):
        if value not in ANNOTATION_OPTIONS:
            return
        self.annotation_var.set(value)

    def _clear_annotation(self):
        component_name = self._current_component_name()
        if component_name is None:
            return
        self.annotations.pop(component_name, None)
        self._annotation_update_in_progress = True
        self.annotation_var.set("")
        self._annotation_update_in_progress = False
        self._display_current()

    def _export_annotations(self):
        annotation_table = _build_annotation_export_table(
            component_table=self.component_table,
            figures_by_index=self.figures_by_index,
            annotations=self.annotations,
        )
        _write_annotation_table(annotation_table, self.annotations_out_path)
        non_empty = int(annotation_table[ANNOTATION_COLUMN].notna().sum())
        self.export_status_var.set(
            f"Saved {non_empty} annotations to {self.annotations_out_path}"
        )

    def _on_close(self):
        # Save on close so manual annotation progress is not lost.
        self._export_annotations()
        self.root.destroy()

    def _refresh_sorted_components(self, *, reset_index: bool):
        ascending = not self.descending
        self.sorted_components = _build_sorted_components(
            component_table=self.component_table,
            metric=self.current_metric,
            figures_by_index=self.figures_by_index,
            ascending=ascending,
        )
        if reset_index:
            self.current_idx = 0
        self.current_idx = max(0, min(self.current_idx, max(len(self.sorted_components) - 1, 0)))
        self._display_current()

    def _move(self, step: int):
        if self.sorted_components.empty:
            return
        self.current_idx = (self.current_idx + step) % len(self.sorted_components)
        self._display_current()

    def _display_current(self):
        import tkinter as tk

        if self.sorted_components.empty:
            self.status_var.set("No component figures found for selected metric.")
            self.image_label.configure(image="", text="No matching component figures found.")
            return

        row = self.sorted_components.iloc[self.current_idx]
        metric_val = row["metric_value"]
        if pd.isna(metric_val):
            metric_text = "NaN"
        else:
            metric_text = f"{metric_val:.6g}"

        figure_path = Path(row["figure_path"])
        self._photo = tk.PhotoImage(file=str(figure_path))
        self.image_label.configure(image=self._photo, text="")
        component_name = str(row["Component"])
        annotation = self.annotations.get(component_name, "None")
        current_annotation = self.annotations.get(component_name, "")
        self._annotation_update_in_progress = True
        self.annotation_var.set(current_annotation)
        self._annotation_update_in_progress = False
        n_annotated = sum(
            value in ANNOTATION_OPTIONS for value in self.annotations.values()
        )

        self.status_var.set(
            "Component {component} ({rank}/{total}) | {metric} = {value} | "
            "Annotation: {annotation} | Annotated: {n_annotated} | Figure: {name}".format(
                component=component_name,
                rank=self.current_idx + 1,
                total=len(self.sorted_components),
                metric=self.current_metric,
                value=metric_text,
                annotation=annotation,
                n_annotated=n_annotated,
                name=figure_path.name,
            )
        )


def component_browser_workflow(
    *,
    metrics_tsv: str,
    figures_dir: str = None,
    metric: str = None,
    descending=False,
    annotations_out: str = None,
    annotations_in: str = None,
):
    """Launch interactive component browser GUI."""
    import tkinter as tk

    metrics_path = Path(metrics_tsv).expanduser().resolve()
    if not metrics_path.exists():
        raise ValueError(f"Metrics TSV does not exist: {metrics_path}")

    if figures_dir is None:
        figures_path = metrics_path.parent / "figures"
    else:
        figures_path = Path(figures_dir).expanduser().resolve()
    if not figures_path.exists():
        raise ValueError(f"Figures directory does not exist: {figures_path}")

    if annotations_out is None:
        annotations_out_path = metrics_path.parent / "slice_artifact_annotations.tsv"
    else:
        annotations_out_path = Path(annotations_out).expanduser().resolve()

    if annotations_in is not None:
        annotations_in_path = Path(annotations_in).expanduser().resolve()
    elif annotations_out_path.exists():
        annotations_in_path = annotations_out_path
    else:
        annotations_in_path = None

    run_prefix = _get_run_prefix(metrics_path)

    component_table = pd.read_table(metrics_path)
    if "Component" not in component_table.columns:
        raise ValueError("Metrics TSV must include a 'Component' column.")

    metric_options = _get_numeric_metric_columns(component_table)
    if not metric_options:
        raise ValueError("No numeric metrics found in TSV.")

    if metric is None:
        metric = metric_options[0]
    elif metric not in metric_options:
        raise ValueError(
            f"Metric '{metric}' not found or not numeric. "
            f"Available metrics: {', '.join(metric_options)}"
        )

    figures_by_index = _collect_component_figures(figures_path, run_prefix=run_prefix)
    if not figures_by_index:
        raise ValueError(
            "No component figures matching run prefix '{}' found in {}".format(
                run_prefix,
                figures_path,
            )
        )

    existing_annotations = {}
    if annotations_in_path is not None:
        existing_annotations = _load_existing_annotations(annotations_in_path)

    root = tk.Tk()
    _ComponentBrowserApp(
        root,
        component_table=component_table,
        metrics=metric_options,
        figures_by_index=figures_by_index,
        initial_metric=metric,
        descending=descending,
        annotations_out_path=annotations_out_path,
        existing_annotations=existing_annotations,
    )
    root.mainloop()


def _main(argv=None):
    """Entry point for command line use."""
    parser = _get_parser()
    args = parser.parse_args(argv)
    component_browser_workflow(
        metrics_tsv=args.metrics_tsv,
        figures_dir=args.figures_dir,
        metric=args.metric,
        descending=args.descending,
        annotations_out=args.annotations_out,
        annotations_in=args.annotations_in,
    )


if __name__ == "__main__":
    _main()
