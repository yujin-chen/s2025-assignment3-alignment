import matplotlib.pyplot as plt
import json
from common import KOA_PATH, DATA_PATH, BEFORE_FINETUNED_RESULT_PATH, FINETUNED_RESULT_PATH, DPO_RESULT_PATH

# Plotting function for loss curves
def plot_loss_curve():
    # Load the metrics from the JSON file
    with open(str(KOA_PATH / "Qwen2.5-finetuned/metrics.json")) as f:
        metrics = json.load(f)
    fig, ax1 = plt.subplots()

    # Plot training loss on primary Y-axis
    ax1.plot(metrics["steps"], metrics["training_losses"], label="Train Loss", color='tab:blue')
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Train Loss", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot validation loss on secondary Y-axis
    val_steps = list(range(500, 500 * len(metrics["validation_losses"]) + 1, 500))
    ax2 = ax1.twinx()
    ax2.plot(val_steps, metrics["validation_losses"], label="Val Loss", color='tab:orange')
    ax2.set_ylabel("Val Loss", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Title and layout
    plt.title("Instruction Fine-tuning Loss Curve")
    fig.tight_layout()
    plt.savefig( str(FINETUNED_RESULT_PATH / "finetuned_loss_curve_dual_axis.png"))
    plt.show()

def plot_accuracy_curve():
    # Load the metrics from the JSON file
    with open(str(KOA_PATH / "qwen2.5_dpo/metrics.json")) as f:
        data = json.load(f)

    # Extract validation accuracy
    val_accuracy = data["val_accuracy"]
    steps = [x[0] for x in val_accuracy]
    accs = [x[1] for x in val_accuracy]

    # plot validation accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(steps, accs, marker='o', label='Validation Accuracy')
    plt.title("Validation Accuracy over Training Steps")
    plt.xlabel("Step")
    plt.ylabel("Validation Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    # save the plot
    plt.savefig(str( DPO_RESULT_PATH / "DPO_val_accuracy_plot.png"))
    print("Saved as val_accuracy_plot.png")


if __name__ == "__main__":
    # plot_loss_curve()
    plot_accuracy_curve()
