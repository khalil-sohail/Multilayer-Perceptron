import math
import matplotlib.pyplot as plt


def plot_history(history: dict[str, list[float]]) -> None:
    """
    Plots the training and validation metrics over the course of training.

    Args:
        history (dict[str, list[float]]): A dictionary containing the training and validation metrics.
    """
    metrics = [
        key
        for key in history
        if not key.startswith("val_")
        and f"val_{key}" in history
    ]

    cols = 2
    rows = math.ceil(len(metrics) / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(8, 3 * rows),
    )

    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        val_metric = f"val_{metric}"
        epochs = range(1, len(history[metric]) + 1)

        ax.plot(
            epochs,
            history[metric],
            label=f"Training",
        )

        ax.plot(
            epochs,
            history[val_metric],
            label=f"Validation",
            linestyle="--",
        )

        ax.set_xlabel("Epoch")
        ax.set_title(metric.replace("_", " ").title())
        ax.legend()
        ax.grid(True)

    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()
