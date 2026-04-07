from pathlib import Path 
import torch 

PROJECT_ROOT = Path(__file__).resolve().parent.parent 

OUTPUT_BASE = PROJECT_ROOT / "outputs" 
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
LOG_BASE = PROJECT_ROOT/ "logs"
for folder in [OUTPUT_BASE, LOG_BASE, NOTEBOOK_DIR]: 
    folder.mkdir(parents=True, exist_ok=True)


# choose model_key from choices   
available_choice = ['distilbert', 'bert', 'roberta']  
 
MODEL_KEY = "distilbert"
 
if MODEL_KEY == 'distilbert': 
    OUTPUT_DIR = OUTPUT_BASE /"distilbert"
    PREDICTION_DF_NAME = "data/distilbert_pred_df.csv"
    TEST_METRICS_FILE = "distilbert_test_metrics.json"
    MODEL_TOKENIZER = "distilbert-base-uncased"
    MODEL_NAME = "distilbert"
    LOG_FILE = LOG_BASE/ "distilbert_log.log"   

elif MODEL_KEY == "bert": 
    OUTPUT_DIR = OUTPUT_BASE /"bert"
    PREDICTION_DF_NAME = "data/bert_pred_df.csv"
    TEST_METRICS_FILE = "bert_test_metrics.json"
    MODEL_TOKENIZER = "google-bert/bert-base-uncased"
    MODEL_NAME = "bert"
    LOG_FILE = LOG_BASE/"bert_log.log"   
    
else:
    OUTPUT_DIR = OUTPUT_BASE /"roberta"
    PREDICTION_DF_NAME = "data/roberta_pred_df.csv"
    TEST_METRICS_FILE = "roberta_test_metrics.json"
    MODEL_TOKENIZER = "FacebookAI/roberta-base"
    MODEL_NAME = "roberta"
    LOG_FILE = LOG_BASE/ "roberta_log.log" 
     

BEST_MODEL_PATH = OUTPUT_DIR/"best_model"
DATA_PATH = "data/sample_tweet.csv"
TEST_DATA_PATH = "data/test_tweet.csv"

BATCH_SIZE = 32
NUM_EPOCHS = 20 
LEARNING_RATE = 1e-5
LR_TYPE = "linear" 
MAX_SEQ_LEN = 40  # based on the EDA for the number of words in the dataset
WEIGHT_DECAY = 0.01


SEED = 777
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COL = "Text"
LABEL_COL = "Target"
ENCODING = "ISO-8859-1"
NUM_LABELS = 2

REQUIRED_CHECKPOINT_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state.pth",
}


