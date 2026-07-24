import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\adity\FSW_Defect_Detection\comparison_runs")

OUTPUT = ROOT / "comparison_plots"
OUTPUT.mkdir(exist_ok=True)

METRICS = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)"
]

COLORS = {
    "yolov5n": "tab:blue",
    "yolov5s": "tab:orange",
    "yolov8n": "tab:green",
    "yolov8s": "tab:red",
    "yolo11n": "tab:purple",
    "yolo11s": "tab:brown",
}

runs = {}

for folder in ROOT.iterdir():

    if not folder.is_dir():
        continue

    csv_file = folder / "results.csv"

    if csv_file.exists():

        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()

        runs[folder.name] = df

print(f"Loaded {len(runs)} runs:")
for r in runs:
    print("  ", r)

for metric in METRICS:

    plt.figure(figsize=(8,6))

    for name, df in runs.items():

        if metric not in df.columns:
            print(f"{metric} missing in {name}")
            continue

        x = range(len(df))

        plt.plot(
            x,
            df[metric],
            linewidth=2,
            label=name,
            color=COLORS.get(name, None)
        )

    plt.title(metric, fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.legend()

    safe_name = (
        metric.replace("/", "_")
              .replace("(", "")
              .replace(")", "")
    )

    plt.tight_layout()
    plt.savefig(OUTPUT / f"{safe_name}.png", dpi=300)
    plt.close()

fig, axes = plt.subplots(2, 5, figsize=(24, 10))
axes = axes.flatten()

for ax, metric in zip(axes, METRICS):

    for name, df in runs.items():

        if metric not in df.columns:
            continue

        x = range(len(df))

        ax.plot(
            x,
            df[metric],
            linewidth=2,
            label=name,
            color=COLORS.get(name, None)
        )

    ax.set_title(metric, fontsize=11)
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)

    
    if ax in [axes[0], axes[5]]:
        ax.set_ylabel(metric)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=len(runs),
    fontsize=11,
    bbox_to_anchor=(0.5, 1.02)
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(
    OUTPUT / "comparison_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nDone!")
print("Graphs saved to:", OUTPUT)