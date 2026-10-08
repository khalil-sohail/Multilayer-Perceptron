import matplotlib.pyplot as plt

def plot_history(histories: dict[str, dict[str, list[float]]]):
    metrics = [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    axes = axes.flatten()
    for index, metric in enumerate(metrics):
        ax = axes[index]

        for optimizer, history in histories.items():
            if metric in history:
                epochs = range(1, len(history[metric]) + 1)

                ax.plot(
                    epochs,
                    history[metric],
                    label=(
                        f"{optimizer.upper()} Training"
                    ),
                )

            val_metric = f"val_{metric}"

            if val_metric in history:
                val_epochs = range(1, len(history[val_metric]) + 1)

                ax.plot(
                    val_epochs,
                    history[val_metric],
                    "--",
                    label=(
                        f"{optimizer.upper()} Validation"
                    ),
                )

        ax.set_title(metric.replace("_", " ").title())

        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)

    for index in range(len(metrics), len(axes)):
        axes[index].set_visible(False)

    plt.tight_layout()
    plt.show()
