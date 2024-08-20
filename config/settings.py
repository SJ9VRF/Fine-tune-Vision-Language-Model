# Configuration settings, like paths and hyperparameters

# Path settings
DATA_DIR = '/path/to/datasets'
IMAGE_DIR = f'{DATA_DIR}/images'
QUESTION_JSON = f'{DATA_DIR}/v2_OpenEnded_mscoco_val2014_questions.json'
ANNOTATION_JSON = f'{DATA_DIR}/v2_mscoco_val2014_annotations.json'
MODEL_SAVE_PATH = '/path/to/saved_model/model.pth'

# Model configuration
MODEL_NAME = 'dandelin/vilt-b32-mlm'
TOKENIZER_NAME = 'dandelin/vilt-b32-mlm'

# Training settings
NUM_EPOCHS = 50
LEARNING_RATE = 5e-5
BATCH_SIZE = 4

# Image processing settings
IMAGE_SIZE = (224, 224)  # Typical size for VILT inputs
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


# Hardware settings
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Evaluation settings
TOP_K_ANSWERS = 5

# Other settings
VERBOSE_LOGGING = True

"""
from config.settings import BATCH_SIZE

# Use BATCH_SIZE in your DataLoader
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

"""
