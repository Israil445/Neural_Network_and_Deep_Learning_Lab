import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from datetime import datetime
import logging

from configs.config import Config
from models.yolov1 import YOLOv1
from models.loss import YOLOLoss
from utils.dataset import WIDERFACEDataset
from utils.visualize import plot_training_history


def setup_logger():
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    log_file = os.path.join(Config.LOGS_DIR, f"train_{datetime.now():%Y%m%d_%H%M%S}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, logger):
    model.train()
    total_loss = 0

    for images, targets in tqdm(dataloader, desc=f"Epoch {epoch} [Train]"):
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {epoch} Training Loss: {avg_loss:.4f}")
    return avg_loss


def validate_one_epoch(model, dataloader, criterion, device, epoch, logger):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc=f"Epoch {epoch} [Val]"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {epoch} Validation Loss: {avg_loss:.4f}")
    return avg_loss


def save_model_checkpoint(model, optimizer, epoch, train_loss, val_loss, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }, filename)


def main():
    logger = setup_logger()
    device = torch.device(Config.DEVICE)

    os.makedirs(Config.WEIGHTS_DIR, exist_ok=True)

    logger.info(f"Using device: {device}")
    logger.info("Loading dataset...")

    full_dataset = WIDERFACEDataset(
        images_path=Config.TRAIN_IMAGES_PATH,
        annotations_path=Config.TRAIN_ANNOTATIONS,
        grid_size=Config.GRID_SIZE,
        input_size=Config.INPUT_SIZE,
        is_train=True
    )

    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = YOLOv1(Config.GRID_SIZE, Config.NUM_BOXES, Config.NUM_CLASSES).to(device)
    criterion = YOLOLoss(Config.GRID_SIZE, Config.NUM_BOXES, Config.NUM_CLASSES, Config.LAMBDA_COORD, Config.LAMBDA_NOOBJ)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    logger.info("Starting training...")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger)
        val_loss = validate_one_epoch(model, val_loader, criterion, device, epoch, logger)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_checkpoint(model, optimizer, epoch, train_loss, val_loss, os.path.join(Config.WEIGHTS_DIR, 'best_model.pth'))
            logger.info(f"Best model saved at epoch {epoch} with Val Loss: {val_loss:.4f}")

        if epoch % 10 == 0 or epoch == Config.NUM_EPOCHS:
            save_model_checkpoint(model, optimizer, epoch, train_loss, val_loss, os.path.join(Config.WEIGHTS_DIR, f'checkpoint_epoch_{epoch}.pth'))

    save_model_checkpoint(model, optimizer, Config.NUM_EPOCHS, train_losses[-1], val_losses[-1], os.path.join(Config.WEIGHTS_DIR, 'final_model.pth'))
    logger.info("Training completed.")

    plot_training_history(train_losses, val_losses)


if __name__ == "__main__":
    main()
