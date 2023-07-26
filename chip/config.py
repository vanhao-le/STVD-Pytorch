
# ========================= Pre-trained ==========================
# IMAGE_SIZE = 224
IMAGE_SIZE = 299 #InceptionV3
BATCH_SIZE = 750



# ========================= Retrain with Siamese ==========================

image_train_dir = r'E:\STVD_DL\data\train'
image_val_dir = r'E:\STVD_DL\data\val'
image_test_dir = r'E:\STVD_DL\data\test'

file_categories = r'data\sub-category.csv'

annotation_train = r'training_data\train.csv'
annotation_val = r'training_data\val.csv'
annotation_test = r'training_data\test.csv'

# ========================= Model Configs ==========================
# image_w = 192
# image_h = 144
# image_mean = [0.485, 0.456, 0.406] 
# image_std = [0.229, 0.224, 0.225]
# image_mean = [0.5, 0.5, 0.5] 
# image_std = [0.229, 0.224, 0.225]
# margin = 1.
# ========================= Learning Configs ==========================
# batch_size = 64
# batch_size = 256
# num_workers = 2
# learning_rate = 1e-2
# momentum = 0.9
# weight_decay = 5e-4
# weight_decay = 1e-3
# step_size = 5
# num_epochs = 50
# embedding_size = 2048