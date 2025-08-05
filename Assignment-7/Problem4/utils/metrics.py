import numpy as np
import torch


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter_area = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0



def evaluate_model(model, dataloader, device, iou_threshold=0.5, confidence_threshold=0.5):
    """Evaluate object detection model using precision, recall, and F1-score."""
    model.eval()

    total_preds = 0
    total_gts = 0
    true_positives = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)

            for img_idx in range(images.size(0)):
                pred_boxes = extract_boxes_from_prediction(preds[img_idx].cpu().numpy(), confidence_threshold)
                gt_boxes = extract_boxes_from_target(targets[img_idx].cpu().numpy())

                total_preds += len(pred_boxes)
                total_gts += len(gt_boxes)

                for p_box in pred_boxes:
                    if any(calculate_iou(p_box, gt_box) > iou_threshold for gt_box in gt_boxes):
                        true_positives += 1

    precision = true_positives / total_preds if total_preds else 0
    recall = true_positives / total_gts if total_gts else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': true_positives,
        'total_predictions': total_preds,
        'total_targets': total_gts
    }



def extract_boxes_from_prediction(pred, confidence_threshold):
    """Extract predicted bounding boxes from YOLO model output."""
    boxes = []
    grid_size = pred.shape[0]

    for i in range(grid_size):
        for j in range(grid_size):
            for box_num in range(2):
                start = box_num * 5
                conf = pred[i, j, start + 4]
                if conf > confidence_threshold:
                    x, y, w, h = pred[i, j, start:start + 4]
                    center_x = (j + x) / grid_size
                    center_y = (i + y) / grid_size
                    boxes.append([
                        center_x - w / 2,
                        center_y - h / 2,
                        center_x + w / 2,
                        center_y + h / 2
                    ])
    return boxes



def extract_boxes_from_target(target):
    """Extract ground truth bounding boxes from target tensor."""
    boxes = []
    grid_size = target.shape[0]

    for i in range(grid_size):
        for j in range(grid_size):
            if target[i, j, 4] > 0:
                x, y, w, h = target[i, j, :4]
                center_x = (j + x) / grid_size
                center_y = (i + y) / grid_size
                boxes.append([
                    center_x - w / 2,
                    center_y - h / 2,
                    center_x + w / 2,
                    center_y + h / 2
                ])
    return boxes
