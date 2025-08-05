import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=1, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        pred = predictions.view(batch_size, self.grid_size, self.grid_size, -1)

        box1 = pred[:, :, :, :5]   
        box2 = pred[:, :, :, 5:10]
        class_pred = pred[:, :, :, 10:]

        target_boxes = targets[:, :, :, :4]
        target_conf = targets[:, :, :, 4:5]
        target_class = targets[:, :, :, 5:]

        obj_mask = target_conf.squeeze(-1) > 0
        noobj_mask = target_conf.squeeze(-1) == 0

        coord_loss = 0
        conf_loss = 0

        best_iou = self.calculate_iou(box1[:, :, :, :4], target_boxes)
        alt_iou = self.calculate_iou(box2[:, :, :, :4], target_boxes)
        responsible_mask = (alt_iou > best_iou).float()
        best_iou = torch.max(best_iou, alt_iou)

        for idx, box in enumerate([box1, box2]):
            responsible = (responsible_mask == idx).float() * obj_mask

            pred_x, pred_y = box[:, :, :, 0], box[:, :, :, 1]
            pred_w = torch.sqrt(torch.abs(box[:, :, :, 2]) + 1e-6)
            pred_h = torch.sqrt(torch.abs(box[:, :, :, 3]) + 1e-6)

            target_x, target_y = target_boxes[:, :, :, 0], target_boxes[:, :, :, 1]
            target_w = torch.sqrt(target_boxes[:, :, :, 2] + 1e-6)
            target_h = torch.sqrt(target_boxes[:, :, :, 3] + 1e-6)

            coord_loss += self.lambda_coord * torch.sum(responsible * ((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2))
            coord_loss += self.lambda_coord * torch.sum(responsible * ((pred_w - target_w) ** 2 + (pred_h - target_h) ** 2))

            conf_target = target_conf.squeeze(-1)
            conf_loss += torch.sum(responsible * (box[:, :, :, 4] - conf_target) ** 2)

            noobj = noobj_mask + (obj_mask * (1 - responsible))
            conf_loss += self.lambda_noobj * torch.sum(noobj * (box[:, :, :, 4]) ** 2)

        class_loss = torch.sum(obj_mask.unsqueeze(-1) * (class_pred - target_class) ** 2)

        total_loss = (coord_loss + conf_loss + class_loss) / batch_size
        return total_loss

    def calculate_iou(self, boxes1, boxes2):
        b1_x1 = boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2
        b1_y1 = boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2
        b1_x2 = boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2
        b1_y2 = boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2

        b2_x1 = boxes2[:, :, :, 0] - boxes2[:, :, :, 2] / 2
        b2_y1 = boxes2[:, :, :, 1] - boxes2[:, :, :, 3] / 2
        b2_x2 = boxes2[:, :, :, 0] + boxes2[:, :, :, 2] / 2
        b2_y2 = boxes2[:, :, :, 1] + boxes2[:, :, :, 3] / 2

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        union = area1 + area2 - inter_area + 1e-6
        iou = inter_area / union

        return iou
