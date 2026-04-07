from transformers import DataCollatorWithPadding
from utils import compute_metrics, tokenizer
import config 
from seed import set_seed
from data import MakeDataset, read_csv_data, stratified_split
from trainer import ModelTrainer, build_training_args
from model import build_model 
from tweet_logger import build_logger 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.utils.class_weight import compute_class_weight

from pathlib import Path
import time
import re 
import numpy as np
import json
import torch

def is_complete_checkpoint(path: Path) -> bool:
    existing = {p.name for p in path.iterdir() if p.is_file()}
    has_model_file = "model.safetensors" in existing or "pytorch_model.bin" in existing 
    return has_model_file and config.REQUIRED_CHECKPOINT_FILES.issubset(existing)

def find_latest_checkpoint(output_dir):
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    
    checkpoints = []
    for p in output_path.iterdir():
        if p.is_dir() and re.match(r"checkpoint-\d+$", p.name):
            step = int(p.name.split("-")[-1])
            if is_complete_checkpoint(p):
                checkpoints.append((step, p))
            else:
                print(f"Skipping incomplete checkpoint: {p}")
            
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x : x[0])
    print(f"Resuming from: {checkpoints[-1][1]}")
    return str(checkpoints[-1][1]) 



def main(config): 
    global tokenizer
    _ = set_seed()
    
    logger, file_handler, console_handler = build_logger()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    
    logger.info("Reading csv file and splitting data...")
    tweet_df = read_csv_data(
        config.DATA_PATH, 
        config.FEATURE_COL, 
        config.LABEL_COL, 
        encoding=config.ENCODING
        )
    
    try:
        train_df, test_df, val_df = stratified_split(tweet_df) 
        
        logger.info(f"============ DATA SPLITS ============="
            f"\n=== val_df ===\n{val_df[config.LABEL_COL].value_counts()}"
            f"\n=== test_df ===\n{test_df[config.LABEL_COL].value_counts()}"
            f"\n=== train_df ===\n{train_df[config.LABEL_COL].value_counts()}"
            )
  
    except Exception as e: 
        logger.critical(f"A critical error occurred:\n\tNO CSV DATA RETREIVED")
        logger.exception(f"An exception occurred as {e}")
        logger.removeHandler(console_handler)
        logger.removeHandler(file_handler)
        return 
    
    logger.debug("Writing test tweet dataset to folder")
    test_df.to_csv("data/test_tweet.csv")
    
    logger.debug("Computing class_weights...")
    labels = train_df[config.LABEL_COL].to_numpy()
    
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    logger.info(f"Computed class weights: \n{class_weights}")
    
    logger.debug("Setting tokenizer and data collator...")
    tokenizer = tokenizer(config.MODEL_TOKENIZER)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    try:
        logger.info("Preparing datasets for modelling...") 
        train_data = MakeDataset(train_df[config.FEATURE_COL], train_df[config.LABEL_COL], tokenizer)
        test_data = MakeDataset(test_df[config.FEATURE_COL], test_df[config.LABEL_COL], tokenizer)
        val_data = MakeDataset(val_df[config.FEATURE_COL], val_df[config.LABEL_COL], tokenizer)
    
    except Exception as e:
        logger.critical(f"A critical error occurred:\n\tDATA PREPARATION FAILED!!!")
        logger.exception(f"An exception occurred as {e}")
        logger.removeHandler(console_handler)
        logger.removeHandler(file_handler)
        return
        
    logger.debug(f"Model selected as >>> {config.MODEL_NAME} <<< ") 
    model = build_model(config.MODEL_NAME, config.NUM_LABELS)

    tuned_model = ModelTrainer(model=model,
                                tokenizer=tokenizer,
                                data_collator=data_collator,
                                compute_metrics=compute_metrics,
                                class_weights = class_weights)
    
    logger.debug("Building training arguments...") 
    training_args= build_training_args()
    trainer = tuned_model.build_trainer(train_data, val_data, training_args)
    
    logger.info("============BATCH TRAINING INFORMATION============")
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Epochs: {config.NUM_EPOCHS}")
    logger.info(f"Steps per epoch: {len(train_data) // config.BATCH_SIZE}")
    logger.info(f"Approx total steps: {(len(train_data) // config.BATCH_SIZE) * config.NUM_EPOCHS}")
    
    latest_checkpoint = find_latest_checkpoint(training_args.output_dir)
    
    if latest_checkpoint is not None:
        logger.info (f"Resuming training from checkpoint: {latest_checkpoint}")
    else: 
        logger.info("No checkpoint found. Starting fresh training run.")
            
    logger.debug("Starting Training...")
    train_start = time.time()
    trainer.train(resume_from_checkpoint=latest_checkpoint)
    train_stop = time.time()
    
    logger.debug("Training Complete") 
    train_time = train_stop - train_start
    logger.info(f"Total train time in seconds: {train_time}")

    logger.debug("Starting Evaluation...")   
    eval_start = time.time()
    trainer.evaluate() 
    eval_stop = time.time()
    
    logger.debug("Evaluation Complete...")
    eval_time = eval_stop - eval_start 
    logger.info(f"Total evaluation time: {eval_time}")
    
    best_model_path = f"{training_args.output_dir}/best_model"
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    
    logger.info("Model training completed successfully with best model saved")
    
    logger.debug("Starting test prediction...")
    test_output = trainer.predict(test_data)
    
    y_pred = np.argmax(test_output.predictions, axis=1)
    y_true = test_output.label_ids 
    
    cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred)
    
    print(cm)
    print(clf_report)
    
    logger.info(f"Confusion matrix: \n{cm}")
    logger.info(f"Classification report: \n{clf_report}")
    logger.info(f"Test metrics: {test_output.metrics}")
    
    combined_results = {
        "test metrics" : test_output.metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report
    }
    with open(config.TEST_METRICS_FILE, "w") as f:
        json.dump(combined_results, f, indent=2)    
    
    logger.info("Model prediction completed succesfully.")    
    logger.removeHandler(console_handler)
    logger.removeHandler(file_handler)
    
    
if __name__ == "__main__":
    main(config)






