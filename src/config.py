from pathlib import Path 
import torch 

PROJECT_ROOT = Path(__file__).resolve().parent.parent 

OUTPUT_DIR = PROJECT_ROOT / "outputs"/"roberta" # modify the path for each model
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
LOG_DIR = PROJECT_ROOT/ "logs"
for folder in [OUTPUT_DIR, LOG_DIR, NOTEBOOK_DIR]: 
    folder.mkdir(parents=True, exist_ok=True)
    

BEST_MODEL_PATH = OUTPUT_DIR/"best_model"
    
DATA_PATH = "data/sample_tweet.csv"
TEST_DATA_PATH = "data/test_tweet.csv"
PREDICTION_DF_NAME = "data/roberta_pred_df.csv"   # update name as per pretrained model name
TEST_METRICS_FILE = "distilbert_test_metrics.json"      # update filename for pretrained model name
    
BATCH_SIZE = 32
NUM_EPOCHS = 20 
LEARNING_RATE = 1e-5
LR_TYPE = "linear" 
MAX_SEQ_LEN = 30  # based on the EDA for the number of words in the dataset
WEIGHT_DECAY = 0.01
MODEL_TOKENIZER = "distilbert-base-uncased" # choose roberta or bert model
NUM_LABELS=2
MODEL_NAME = "distilBERT" # choose BERT or roberta

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


