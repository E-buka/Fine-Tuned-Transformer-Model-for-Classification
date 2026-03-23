from pathlib import Path 
import torch 

PROJECT_ROOT = Path(__file__).resolve().parent.parent 

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
PLOT_DIR = PROJECT_ROOT / "plots"

for folder in [ARTIFACTS_DIR, MODEL_DIR, LOG_DIR, PLOT_DIR]: 
    folder.mkdir(parents=True, exist_ok=True)
    
BATCH_SIZE = 64
NUM_EPOCHS = 2 
LEARNING_RATE = 1e-3 
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 512
WEIGHT_DECAY = 0.01
MODEL_TOKENIZER = "distilbert-base-uncased"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


FEATURE_COL = "Text"
LABEL_COL = "Target"
ENCODING = "ISO-8859-1"

SEED = 777