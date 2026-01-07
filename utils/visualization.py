import matplotlib.pyplot as plt

# Plot training and validation losses
def plot_loss_curves(train_losses, val_losses, config):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, config.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, config.epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{config.loss_curve_save_path}.png')  # Saves the plot as an image file
    plt.show()