import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_predictions(image, predictions, grid_size=7, confidence_threshold=0.5, num_boxes=2):
    """
    Visualize YOLO predictions on an image.
    """
    image = image.copy()
    h, w = image.shape[:2]

    for row in range(grid_size):
        for col in range(grid_size):
            for box_idx in range(num_boxes):
                idx = box_idx * 5
                x, y, box_w, box_h, conf = predictions[row, col, idx:idx + 5]

                if conf > confidence_threshold:
                    center_x = (col + x) / grid_size * w
                    center_y = (row + y) / grid_size * h
                    bw = box_w * w
                    bh = box_h * h

                    x1 = int(center_x - bw / 2)
                    y1 = int(center_y - bh / 2)
                    x2 = int(center_x + bw / 2)
                    y2 = int(center_y + bh / 2)

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f'{conf:.2f}', (x1, max(y1 - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return image



def plot_training_history(train_losses, val_losses=None):
    """
    Plot training and optional validation loss over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)

    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss', linewidth=2, linestyle='--')

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training History', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()