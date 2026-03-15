import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))


def load_metrics(metrics_path="logs/metrics.csv"):
    if not Path(metrics_path).exists():
        print(f"Metrics file not found at {metrics_path}")
        return None
    return pd.read_csv(metrics_path)


def plot_metrics():
    df = load_metrics()
    if df is None:
        return

    output_dir = Path("logs/plots")
    output_dir.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Train vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(df['epoch'], df['train_token_acc'], label='Train Token Accuracy', marker='o')
    ax2.plot(df['epoch'], df['val_token_acc'], label='Val Token Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Token Accuracy')
    ax2.set_title('Train vs Validation Token Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(df['epoch'], df['val_exact_match'], marker='o', color='green')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Exact Match')
    ax3.set_title('Validation Exact Match')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(df['epoch'], df['val_edit_distance'], label='Raw Edit Distance', marker='o')
    ax4.plot(df['epoch'], df['val_norm_edit_distance'], label='Normalized Edit Distance', marker='s')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Edit Distance')
    ax4.set_title('Validation Edit Distance (Levenshtein)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(df['epoch'], df['learning_rate'], marker='o', color='purple')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Learning Rate')
    ax5.set_title('Learning Rate per Epoch')
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(df['epoch'], df['grad_norm'], marker='o', color='red')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Gradient Norm')
    ax6.set_title('Average Gradient Norm per Epoch')
    ax6.grid(True, alpha=0.3)

    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(df['epoch'], df['epoch_time_sec'], marker='o', color='orange')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Time (seconds)')
    ax7.set_title('Epoch Time')
    ax7.grid(True, alpha=0.3)

    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(df['epoch'], df['gpu_memory_mb'], label='Current', marker='o')
    ax8.plot(df['epoch'], df['gpu_memory_peak_mb'], label='Peak', marker='s')
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Memory (MB)')
    ax8.set_title('GPU Memory Usage')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(df['epoch'], df['train_loss'], label='Train Loss', alpha=0.7)
    ax9.plot(df['epoch'], df['val_loss'], label='Val Loss', alpha=0.7)
    ax9.plot(df['epoch'], df['val_exact_match'], label='Val Exact Match', alpha=0.7)
    ax9.plot(df['epoch'], df['val_token_acc'], label='Val Token Acc', alpha=0.7)
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Value')
    ax9.set_title('All Key Metrics Overview')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "training_metrics.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Saved combined plot to {output_path}")

    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train', marker='o')
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val', marker='s')
    axes[0, 0].set_title('Loss Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df['epoch'], df['train_token_acc'], label='Train', marker='o')
    axes[0, 1].plot(df['epoch'], df['val_token_acc'], label='Val', marker='s')
    axes[0, 1].set_title('Token Accuracy Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df['epoch'], df['val_exact_match'], marker='o', color='green')
    axes[1, 0].set_title('Validation Exact Match')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Exact Match Rate')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df['epoch'], df['val_edit_distance'], label='Raw', marker='o')
    axes[1, 1].plot(df['epoch'], df['val_norm_edit_distance'], label='Normalized', marker='s')
    axes[1, 1].set_title('Validation Edit Distance')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Distance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path2 = output_dir / "evaluation_metrics.png"
    plt.savefig(output_path2, dpi=100, bbox_inches='tight')
    print(f"Saved evaluation metrics plot to {output_path2}")

    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))

    axes3[0, 0].plot(df['epoch'], df['learning_rate'], marker='o', color='purple')
    axes3[0, 0].set_title('Learning Rate Schedule')
    axes3[0, 0].set_xlabel('Epoch')
    axes3[0, 0].set_ylabel('Learning Rate')
    axes3[0, 0].grid(True, alpha=0.3)

    axes3[0, 1].plot(df['epoch'], df['grad_norm'], marker='o', color='red')
    axes3[0, 1].set_title('Gradient Norm per Epoch')
    axes3[0, 1].set_xlabel('Epoch')
    axes3[0, 1].set_ylabel('Gradient Norm')
    axes3[0, 1].grid(True, alpha=0.3)

    axes3[1, 0].plot(df['epoch'], df['epoch_time_sec'], marker='o', color='orange')
    axes3[1, 0].set_title('Training Time per Epoch')
    axes3[1, 0].set_xlabel('Epoch')
    axes3[1, 0].set_ylabel('Time (seconds)')
    axes3[1, 0].grid(True, alpha=0.3)

    axes3[1, 1].plot(df['epoch'], df['gpu_memory_mb'], label='Current', marker='o')
    axes3[1, 1].plot(df['epoch'], df['gpu_memory_peak_mb'], label='Peak', marker='s')
    axes3[1, 1].set_title('GPU Memory Usage')
    axes3[1, 1].set_xlabel('Epoch')
    axes3[1, 1].set_ylabel('Memory (MB)')
    axes3[1, 1].legend()
    axes3[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path3 = output_dir / "training_details.png"
    plt.savefig(output_path3, dpi=100, bbox_inches='tight')
    print(f"Saved training details plot to {output_path3}")

    print("\nMetrics Summary:")
    print(f"Total epochs: {len(df)}")
    print(f"\nBest validation loss: {df['val_loss'].min():.4f} (Epoch {df['val_loss'].idxmin() + 1})")
    print(f"Best validation exact match: {df['val_exact_match'].max():.4f} (Epoch {df['val_exact_match'].idxmax() + 1})")
    print(f"Best validation token accuracy: {df['val_token_acc'].max():.4f} (Epoch {df['val_token_acc'].idxmax() + 1})")
    print(f"Best validation edit distance: {df['val_edit_distance'].min():.4f} (Epoch {df['val_edit_distance'].idxmin() + 1})")

    plt.show()


if __name__ == "__main__":
    plot_metrics()
