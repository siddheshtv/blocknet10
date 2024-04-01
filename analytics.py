import matplotlib.pyplot as plt

def model_analytics(train_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Loss per Training Batch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracies')
    plt.legend()

    plt.tight_layout()
    plt.savefig("analytics.png")
    return "✅ Figure saved successfully"