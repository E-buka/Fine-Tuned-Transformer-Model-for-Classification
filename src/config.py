from pathlib import Path 
import torch 

PROJECT_ROOT = Path(__file__).resolve().parent.parent 

OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
PLOT_DIR = PROJECT_ROOT / "plots"

for folder in [OUTPUT_DIR, LOG_DIR, PLOT_DIR]: 
    folder.mkdir(parents=True, exist_ok=True)
    
    
DATA_PATH = "data/tweets.csv"
    
BATCH_SIZE = 64
NUM_EPOCHS = 20 
LEARNING_RATE = 1e-3 
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 512
WEIGHT_DECAY = 0.01
MODEL_TOKENIZER = "distilbert-base-uncased"
NUM_LABELS=2
MODEL_NAME = "distilBERT"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


FEATURE_COL = "Text"
LABEL_COL = "Target"
ENCODING = "ISO-8859-1"

SEED = 777

REQUIRED_CHECKPOINT_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state.pth",
}


