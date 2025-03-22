class ModelConfig:
    # Input
    INPUT_WIDTH = 608
    INPUT_HEIGHT = 608
    
    # Backbone
    BACKBONE = 'CSPDarknet53'
    
    # Neck
    NECK = 'PANet'
    SPP_KERNELS = [5, 9, 13]
    
    # Head
    NUM_CLASSES = 20  # Pascal VOC classes
    ANCHORS = [
        [(12, 16), (19, 36), (40, 28)],      # P3/8
        [(36, 75), (76, 55), (72, 146)],     # P4/16
        [(142, 110), (192, 243), (459, 401)] # P5/32
    ]
    
    # Training
    BATCH_SIZE = 8
    SUBDIVISIONS = 1  # Subdivide batch to fit in GPU memory
    MAX_EPOCHS = 300
    
    # Optimizer
    LEARNING_RATE = 0.01
    MOMENTUM = 0.937
    WEIGHT_DECAY = 0.0005
    
    # Learning Rate Schedule
    WARMUP_EPOCHS = 3
    
    # Loss weights
    LAMBDA_COORD = 5.0
    LAMBDA_NOOBJ = 0.5
    
    # Thresholds
    CONF_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    
    # Data Augmentation
    MOSAIC_PROB = 0.5
    MIXUP_PROB = 0.15
    CUTMIX_PROB = 0.15
    
    # Advanced parameters
    MISH_ACTIVATION = True
    SYNC_BN = True  # Sync BatchNorm for multi-GPU training
