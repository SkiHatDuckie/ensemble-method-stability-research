# src/evaluate.py
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from metrics import Metric, MetricActions


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "evaluate",
        description="Generate comparison charts from all available results files in the results folder.")
    parser.add_argument("-results-files", nargs="*", default=None,
                        help="Optional list of result files to use. If omitted, all supported results files in the results directory are loaded.")
    parser.add_argument("-results-dir", default="results",
                        help="Directory containing result files.")
    parser.add_argument("-output-dir", default="eval",
                        help="Directory where charts and summary CSVs are saved.")
    parser.add_argument("-debug", action="store_true",
                        help="Print metric summaries for each method and experiment.")
    return parser


def create_results_directory(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_supported_results_files(results_dir: Path) -> list[Path]:
    candidates = []
    candidates.extend(sorted(results_dir.glob("control_*.txt"), key=lambda p: p.stat().st_mtime))
    candidates.extend(sorted(results_dir.glob("noise*_*.txt"), key=lambda p: p.stat().st_mtime))
    candidates.extend(sorted(results_dir.glob("control_raw_*.csv"), key=lambda p: p.stat().st_mtime))
    if not candidates:
        raise FileNotFoundError(f"No supported result files found in {results_dir}")
    return candidates


def parse_noise_percent(path: Path) -> float | None:
    stem = path.stem
    if stem.startswith("control"):
        return 0.0
    match = re.search(r"noise(\d+)", stem)
    if match:
        return float(match.group(1))
    return None


BASELINE_SAMPLE_TYPE = "baseline (no noise injection)"


def format_noise_label(noise_percent: float | None) -> str:
    if noise_percent is None:
        return "unknown noise"
    if noise_percent == 0.0:
        return "control (no noise injection)"
    return f"{noise_percent}% noise"


def normalize_sample_type(sample_type: str | None) -> str:
    if sample_type is None:
        return "unknown"
    normalized = str(sample_type).strip()
    if normalized.lower().startswith("control"):
        return BASELINE_SAMPLE_TYPE
    return normalized


def prepare_results_df(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()
    df["sample_type_label"] = df["sample_type"].apply(normalize_sample_type)
    df["is_baseline_sample_type"] = df["sample_type_label"] == BASELINE_SAMPLE_TYPE
    df["accuracy_gap"] = df["training_accuracy"] - df["testing_accuracy"]
    return df


def remove_baseline_legend(ax: plt.Axes, title: str) -> None:
    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if l != BASELINE_SAMPLE_TYPE]
    if filtered:
        handles, labels = zip(*filtered)
        ax.legend(handles, labels, title=title, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        if ax.legend_:
            ax.legend_.remove()


def parse_txt_result_file(path: Path) -> pd.DataFrame:
    content = path.read_text(encoding="utf-8")
    noise_percent = parse_noise_percent(path)
    header_match = re.search(r"=== Testing with (\d+)% noise ===", content)
    if header_match:
        noise_percent = float(header_match.group(1))

    rows = []
    current_method = None
    current_sample_type = None
    current_values = {}

    def flush_current_row():
        if current_method and current_sample_type and current_values:
            row = {
                "source_file": path.name,
                "experiment": path.stem,
                "noise_percent": noise_percent,
                "method": current_method,
                "sample_type": current_sample_type,
                "training_accuracy": current_values.get("training_accuracy"),
                "testing_accuracy": current_values.get("testing_accuracy"),
                "f1_score": current_values.get("f1_score"),
                "roc_auc": current_values.get("roc_auc"),
                "training_time": current_values.get("training_time"),
            }
            rows.append(row)

    for line in content.splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("Method:"):
            flush_current_row()
            current_method = text.split("Method:", 1)[1].strip()
            current_sample_type = None
            current_values = {}
            continue
        if text.startswith("Sample type:"):
            flush_current_row()
            current_sample_type = text.split("Sample type:", 1)[1].strip()
            current_values = {}
            continue

        if text.startswith("avg.") and ":" in text:
            name, value = text.split(":", 1)
            key = name.lower().strip()
            value = value.strip()
            if value.endswith("%"):
                numeric = float(value[:-1].strip())
            elif value.endswith("sec"):
                numeric = float(value[:-3].strip())
            else:
                numeric = float(value)

            if "training accuracy" in key:
                current_values["training_accuracy"] = numeric
            elif "testing accuracy" in key:
                current_values["testing_accuracy"] = numeric
            elif "testing f1" in key:
                current_values["f1_score"] = numeric
            elif "testing roc-au" in key or "testing roc_auc" in key:
                current_values["roc_auc"] = numeric
            elif "training time" in key:
                current_values["training_time"] = numeric

    flush_current_row()
    if not rows:
        raise ValueError(f"No parsed metric rows found in result file: {path}")
    return pd.DataFrame(rows)


def parse_csv_result_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "method" not in df.columns and "model" in df.columns:
        df = df.rename(columns={"model": "method"})
    if "sample_type" not in df.columns:
        raise ValueError(f"CSV result file is missing sample_type column: {path}")
    noise_percent = parse_noise_percent(path)
    df = df.copy()
    df["source_file"] = path.name
    df["experiment"] = path.stem
    df["noise_percent"] = noise_percent
    return df


def load_all_results(files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in files:
        if path.suffix.lower() == ".txt":
            frames.append(parse_txt_result_file(path))
        elif path.suffix.lower() == ".csv":
            frames.append(parse_csv_result_file(path))
    return pd.concat(frames, ignore_index=True)


def plot_file_charts(results_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("testing_accuracy", "Testing Accuracy"),
        ("f1_score", "F1 Score"),
        ("roc_auc", "ROC-AUC"),
        ("accuracy_gap", "Training - Testing Accuracy"),
    ]
    grouped_by_noise = (
        results_df
        .groupby(["noise_percent", "method", "sample_type_label"], as_index=False)
        .mean(numeric_only=True)
    )
    for noise_percent, group in grouped_by_noise.groupby("noise_percent"):
        noise_label = format_noise_label(noise_percent)
        file_safe = "control" if noise_percent == 0.0 else f"noise_{int(noise_percent) if float(noise_percent).is_integer() else noise_percent}"
        for column, title in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            if noise_percent == 0.0:
                sns.barplot(
                    data=group,
                    x="method",
                    y=column,
                    hue="method",
                    ax=ax,
                    palette="deep",
                    legend=False,
                    errorbar=None,
                )
            else:
                sns.barplot(
                    data=group,
                    x="method",
                    y=column,
                    hue="sample_type_label",
                    ax=ax,
                    palette="deep",
                    errorbar=None,
                )
            ax.set_title(f"{title} by Method for {noise_label}")
            ax.set_ylabel(title)
            ax.set_xlabel("Method")
            if noise_percent != 0.0:
                ax.legend(title="Sample type")
                remove_baseline_legend(ax, title="Sample type")
            else:
                ax.legend_.remove() if ax.legend_ else None
            plt.xticks(rotation=25)
            plt.tight_layout()
            fig.savefig(output_dir / f"{file_safe}_{column}.png", dpi=200)
            plt.close(fig)


def plot_combined_charts(results_df: pd.DataFrame, output_dir: Path) -> None:
    combined = results_df.dropna(subset=["noise_percent"]).copy()
    combined["noise_percent"] = combined["noise_percent"].astype(float)
    combined = (
        combined
        .groupby(["noise_percent", "method", "sample_type_label"], as_index=False)
        .mean(numeric_only=True)
        .sort_values("noise_percent")
    )

    metrics = [
        ("testing_accuracy", "Testing Accuracy"),
        ("f1_score", "F1 Score"),
        ("roc_auc", "ROC-AUC"),
        ("accuracy_gap", "Training - Testing Accuracy"),
    ]

    for column, title in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=combined,
            x="noise_percent",
            y=column,
            hue="method",
            style="sample_type_label",
            markers=True,
            dashes=False,
            ax=ax,
            palette="tab10",
        )
        ax.set_title(f"{title} vs. Noise Percent")
        ax.set_ylabel(title)
        ax.set_xlabel("Percent noise added")
        remove_baseline_legend(ax, title="Method / Sample type")
        plt.tight_layout()
        fig.savefig(output_dir / f"combined_{column}.png", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True)
    for ax, (column, title) in zip(axes, metrics):
        sns.lineplot(
            data=combined,
            x="noise_percent",
            y=column,
            hue="method",
            style="sample_type_label",
            markers=True,
            dashes=False,
            ax=ax,
            palette="tab10",
            legend=(ax is axes[0]),
        )
        ax.set_title(title)
        ax.set_xlabel("Percent noise added")
        ax.set_ylabel(title)
    remove_baseline_legend(axes[0], title="Method / Sample type")
    plt.tight_layout()
    fig.savefig(output_dir / "combined_all_metrics.png", dpi=200)
    plt.close(fig)

    for sample_type, sample_group in combined.groupby("sample_type_label"):
        if sample_type == BASELINE_SAMPLE_TYPE:
            continue
        safe_sample_type = sample_type.replace(" ", "_").replace("/", "_").replace("+", "p")
        for column, title in metrics:
            if sample_group[column].isna().all():
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(
                data=sample_group,
                x="noise_percent",
                y=column,
                hue="method",
                markers=True,
                dashes=False,
                ax=ax,
                palette="tab10",
            )
            ax.set_title(f"{title} vs. Noise Percent for {sample_type}")
            ax.set_ylabel(title)
            ax.set_xlabel("Percent noise added")
            ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            fig.savefig(output_dir / f"combined_{column}_{safe_sample_type}.png", dpi=200)
            plt.close(fig)


def plot_metric_model_by_sample_type(results_df: pd.DataFrame, output_dir: Path) -> None:
    combined = results_df.dropna(subset=["noise_percent"]).copy()
    combined["noise_percent"] = combined["noise_percent"].astype(float)
    combined = (
        combined
        .groupby(["noise_percent", "method", "sample_type_label"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["method", "noise_percent"])
    )

    metrics = [
        ("testing_accuracy", "Testing Accuracy"),
        ("f1_score", "F1 Score"),
        ("roc_auc", "ROC-AUC"),
    ]

    for method, method_group in combined.groupby("method"):
        safe_method = method.replace(" ", "_").replace("/", "_").replace("+", "p")
        method_group = method_group[method_group["sample_type_label"] != BASELINE_SAMPLE_TYPE]
        if method_group.empty:
            continue
        for column, title in metrics:
            if method_group[column].isna().all():
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(
                data=method_group,
                x="noise_percent",
                y=column,
                hue="sample_type_label",
                markers=True,
                dashes=False,
                ax=ax,
                palette="tab10",
            )
            ax.set_title(f"{title} for {method} by Sample Type")
            ax.set_ylabel(title)
            ax.set_xlabel("Percent noise added")
            ax.legend(title="Sample type", bbox_to_anchor=(1.02, 1), loc="upper left")
            remove_baseline_legend(ax, title="Sample type")
            plt.tight_layout()
            fig.savefig(output_dir / f"{safe_method}_{column}_by_sample_type.png", dpi=200)
            plt.close(fig)


def build_metric_summary(results_df: pd.DataFrame) -> list[tuple[str, str, str, list[Metric]]]:
    summary = []
    grouped = results_df.groupby(["method", "sample_type", "experiment"])
    for (method, sample_type, experiment), group in grouped:
        metrics = [
            Metric(name="testing accuracy", actions=[MetricActions.PERCENT_AVERAGE], decimal_precision=2, data=group["testing_accuracy"].tolist()),
            Metric(name="testing F1 score", actions=[MetricActions.AVERAGE], decimal_precision=3, data=group["f1_score"].tolist()),
            Metric(name="testing ROC-AUC", actions=[MetricActions.AVERAGE], decimal_precision=3, data=group["roc_auc"].tolist()),
            Metric(name="training time", actions=[MetricActions.AVERAGE, MetricActions.TOTAL], suffix="sec", decimal_precision=3, data=group["training_time"].tolist()),
        ]
        summary.append((method, sample_type, experiment, metrics))
    return summary


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    output_dir = create_results_directory(args.output_dir)
    results_dir = Path(args.results_dir)

    if args.results_files:
        paths = [Path(path) for path in args.results_files]
    else:
        paths = find_supported_results_files(results_dir)

    results_df = load_all_results(paths)
    results_df["noise_percent"] = results_df["noise_percent"].astype(float)
    results_df = prepare_results_df(results_df)

    summary_path = output_dir / "evaluate_combined_summary.csv"
    results_df.to_csv(summary_path, index=False)

    plot_file_charts(results_df, output_dir)
    plot_combined_charts(results_df, output_dir)
    plot_metric_model_by_sample_type(results_df, output_dir)

    print(f"Loaded {len(paths)} result file(s)")
    print(f"Saved combined summary CSV to: {summary_path}")
    print(f"Saved charts to: {output_dir}")

    if args.debug:
        metric_summary = build_metric_summary(results_df)
        for method, sample_type, experiment, metrics in metric_summary:
            print(f"\nMethod: {method} | Sample type: {sample_type} | Experiment: {experiment}")
            for metric in metrics:
                print(metric, end="")


if __name__ == "__main__":
    main()
