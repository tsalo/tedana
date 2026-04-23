"""Interactive browser for tedana component figures sorted by metric values."""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

LGR = logging.getLogger("GENERAL")


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


def _collect_component_figures(figures_dir: Path) -> Dict[int, Path]:
    """Map component index to static figure path."""
    figures_by_index: Dict[int, Path] = {}
    for file_ in sorted(figures_dir.glob("*comp_*.png")):
        match = re.search(r"comp_(\d+)\.png$", file_.name)
        if not match:
            continue
        comp_idx = int(match.group(1))
        figures_by_index.setdefault(comp_idx, file_)
    return figures_by_index


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

        self.image_label = ttk.Label(root, anchor="center")
        self.image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.root.bind("<Left>", lambda _event: self._move(-1))
        self.root.bind("<Right>", lambda _event: self._move(1))
        self.root.bind("<Up>", lambda _event: self._move(-1))
        self.root.bind("<Down>", lambda _event: self._move(1))
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        self.root.bind("<Button-4>", lambda _event: self._move(-1))
        self.root.bind("<Button-5>", lambda _event: self._move(1))

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

        self.status_var.set(
            "Component {component} ({rank}/{total}) | {metric} = {value} | Figure: {name}".format(
                component=row["Component"],
                rank=self.current_idx + 1,
                total=len(self.sorted_components),
                metric=self.current_metric,
                value=metric_text,
                name=figure_path.name,
            )
        )


def component_browser_workflow(
    *,
    metrics_tsv: str,
    figures_dir: str = None,
    metric: str = None,
    descending=False,
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

    figures_by_index = _collect_component_figures(figures_path)
    if not figures_by_index:
        raise ValueError(f"No component figures matching '*comp_*.png' found in {figures_path}")

    root = tk.Tk()
    _ComponentBrowserApp(
        root,
        component_table=component_table,
        metrics=metric_options,
        figures_by_index=figures_by_index,
        initial_metric=metric,
        descending=descending,
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
    )


if __name__ == "__main__":
    _main()
