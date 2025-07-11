import os
import cv2
import torch
import argparse
import numpy as np

from models.yolov1 import YOLOv1
from configs.config import Config
from utils.visualize import visualize_predictions


def load_yolo_model(weights_path, device):
    model = YOLOv1(
        grid_size=Config.GRID_SIZE,
        num_boxes=Config.NUM_BOXES,
        num_classes=Config.NUM_CLASSES
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def preprocess_image(image_path, input_size):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image_rgb.copy()

    image_resized = cv2.resize(image_rgb, (input_size, input_size)).astype(np.float32) / 255.0
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    image_normalized = (image_resized - mean) / std

    tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float()

    return tensor, original


def non_max_suppression(predictions, conf_thresh=0.5, iou_thresh=0.4):
    boxes, scores = [], []
    grid_size = predictions.shape[0]

    for y in range(grid_size):
        for x in range(grid_size):
            for b in range(2):
                idx = b * 5
                pred = predictions[y, x, idx:idx+5]

                if pred[4] > conf_thresh:
                    cx = (x + pred[0]) / grid_size
                    cy = (y + pred[1]) / grid_size
                    bw, bh = pred[2], pred[3]

                    x1, y1 = cx - bw / 2, cy - bh / 2
                    x2, y2 = cx + bw / 2, cy + bh / 2

                    boxes.append([x1, y1, x2, y2])
                    scores.append(pred[4])

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({'bbox': boxes[i], 'confidence': scores[i]})

    return results


def detect_and_visualize(model, image_path, device, conf_thresh, save_path=None):
    tensor, original = preprocess_image(image_path, Config.INPUT_SIZE)
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor).squeeze(0).cpu().numpy()

    detections = non_max_suppression(output, conf_thresh)

    img_h, img_w = original.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        x1, y1, x2, y2 = int(x1 * img_w), int(y1 * img_h), int(x2 * img_w), int(y2 * img_h)
        cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original, f"{det['confidence']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = save_path if save_path else 'output_face_detection.jpg'
    cv2.imwrite(output_path, cv2.cvtColor(original, cv2.COLOR_RGB2BGR))

    return detections, output_path


def main():
    parser = argparse.ArgumentParser(description='YOLOv1 Face Detection')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--weights', type=str, default='outputs/weights/best_face_detector_yolov1.pth', help='Model weights path')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', type=str, help='Output image path')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')

    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    model = load_yolo_model(args.weights, device)

    detections, out_path = detect_and_visualize(model, args.image, device, args.confidence, args.output)

    print(f"Detections saved to: {out_path}")
    print(f"Total faces detected: {len(detections)}")

    for idx, det in enumerate(detections, 1):
        print(f"Face {idx}: Confidence = {det['confidence']:.3f}")


if __name__ == "__main__":
    main()


'''
run code to detect:    python detect.py --image person.jpg --weights outputs/weights/best_model.pth --output detected_person.jpg

'''