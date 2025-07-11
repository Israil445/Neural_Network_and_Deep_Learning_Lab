import torch

class Config:
    # Dataset paths
    DATASET_ROOT = "data/WIDERFACE"
    TRAIN_IMAGES_PATH = "data/WIDERFACE/WIDER_train/images"
    VAL_IMAGES_PATH = "data/WIDERFACE/WIDER_val/images"
    TRAIN_ANNOTATIONS = "data/WIDERFACE/wider_face_split/wider_face_train_bbx_gt.txt"
    VAL_ANNOTATIONS = "data/WIDERFACE/wider_face_split/wider_face_val_bbx_gt.txt"
    
    # Model parameters
    INPUT_SIZE = 448
    GRID_SIZE = 7
    NUM_BOXES = 2
    NUM_CLASSES = 1 
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 50
    WEIGHT_DECAY = 0.0005
    
    # Loss weights
    LAMBDA_COORD = 5
    LAMBDA_NOOBJ = 0.5
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    WEIGHTS_DIR = "outputs/weights"
    LOGS_DIR = "outputs/logs"
    PREDICTIONS_DIR = "outputs/predictions"