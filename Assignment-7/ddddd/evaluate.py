import os
import torch
import argparse
from torch.utils.data import DataLoader

from configs.yolov1_config import Config
from models.yolov1 import YOLOv1
from utils.dataset import WIDERFACEDataset
from utils.metrics import evaluate_model


def load_model(weights_path, device):
    model = YOLOv1(
        grid_size=Config.GRID_SIZE,
        num_boxes=Config.NUM_BOXES,
        num_classes=Config.NUM_CLASSES
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def load_dataset(split):
    if split == 'val':
        return WIDERFACEDataset(
            images_path=Config.VAL_IMAGES_PATH,
            annotations_path=Config.VAL_ANNOTATIONS,
            grid_size=Config.GRID_SIZE,
            input_size=Config.INPUT_SIZE,
            is_train=False
        )
    else:
        raise NotImplementedError(f"Dataset split '{split}' not implemented.")


def evaluate(weights, data_split, confidence_thresh, iou_thresh, device):
    print(f"Loading model from {weights}...")
    model = load_model(weights, device)

    dataset = load_dataset(data_split)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    print(f"Evaluating on {len(dataset)} images with confidence>{confidence_thresh} and IoU>{iou_thresh}...")

    metrics = evaluate_model(model, dataloader, device, iou_threshold=iou_thresh, confidence_threshold=confidence_thresh)

    print("\nEvaluation Results:")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Recall:          {metrics['recall']:.4f}")
    print(f"F1 Score:        {metrics['f1_score']:.4f}")
    print(f"True Positives:  {metrics['true_positives']}")
    print(f"Total Predictions: {metrics['total_predictions']}")
    print(f"Total Targets:   {metrics['total_targets']}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv1 Face Detector Evaluation")
    parser.add_argument('--weights', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--data', type=str, default='val', choices=['val'], help='Dataset split')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    evaluate(args.weights, args.data, args.confidence, args.iou, device)


if __name__ == "__main__":
    main()
