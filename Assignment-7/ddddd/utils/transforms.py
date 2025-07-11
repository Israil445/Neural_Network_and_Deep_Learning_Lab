import albumentations as A


def get_train_transforms(input_size=448):
    """
    Data augmentation pipeline for training images and bounding boxes.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.Blur(blur_limit=3, p=0.1),
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))



def get_val_transforms(input_size=448):
    """
    Validation pipeline without random augmentations.
    """
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
