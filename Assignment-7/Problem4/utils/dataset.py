import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A

class WIDERFACEDataset(Dataset):
    def __init__(self, images_path, annotations_path, grid_size=7, input_size=448, is_train=True):
        self.images_path = images_path
        self.grid_size = grid_size
        self.input_size = input_size
        self.is_train = is_train

        self.image_files, self.annotations = self._parse_annotations(annotations_path)
        self.transform = self._build_transform()

    def _build_transform(self):
        transform_list = [A.Resize(self.input_size, self.input_size),
                          A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        if self.is_train:
            transform_list = [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3)
            ] + transform_list

        return A.Compose(transform_list, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def _parse_annotations(self, annotations_path):
        image_files, annotations = [], []

        with open(annotations_path, 'r') as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.endswith('.jpg'):
                img_path = line
                i += 1

                try:
                    num_faces = int(lines[i].strip())
                except (IndexError, ValueError):
                    i += 1
                    continue

                i += 1
                faces = []

                for _ in range(num_faces):
                    if i < len(lines):
                        bbox_info = lines[i].strip().split()
                        if len(bbox_info) >= 4:
                            try:
                                x, y, w, h = map(int, bbox_info[:4])
                                if w > 10 and h > 10 and x >= 0 and y >= 0:
                                    faces.append([x, y, w, h])
                            except ValueError:
                                pass
                        i += 1

                if faces:
                    image_files.append(img_path)
                    annotations.append(faces)
            else:
                i += 1

        return image_files, annotations

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_path, self.image_files[idx])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        bboxes, class_labels = [], []

        for x, y, bw, bh in self.annotations[idx]:
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            bw_norm, bh_norm = bw / w, bh / h

            bboxes.append([np.clip(cx, 0, 1), np.clip(cy, 0, 1), np.clip(bw_norm, 0, 1), np.clip(bh_norm, 0, 1)])
            class_labels.append(0)

        if bboxes:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image, bboxes = transformed['image'], transformed['bboxes']
        else:
            fallback_transform = A.Compose([
                A.Resize(self.input_size, self.input_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = fallback_transform(image=image)['image']

        image = torch.from_numpy(image).permute(2, 0, 1).float()

        target = torch.zeros((self.grid_size, self.grid_size, 6))

        for bbox in bboxes:
            cx, cy, bw, bh = bbox
            grid_x = min(int(cx * self.grid_size), self.grid_size - 1)
            grid_y = min(int(cy * self.grid_size), self.grid_size - 1)

            x_cell, y_cell = cx * self.grid_size - grid_x, cy * self.grid_size - grid_y

            target[grid_y, grid_x, :4] = torch.tensor([x_cell, y_cell, bw, bh])
            target[grid_y, grid_x, 4] = 1.0
            target[grid_y, grid_x, 5] = 1.0

        return image, target
