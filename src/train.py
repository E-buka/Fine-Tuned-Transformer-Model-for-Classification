from transformers import DataCollatorWithPadding
from utils import compute_metrics
import config 
from seed import set_seed
from utils import tokenizer 
from data import MakeDataset, read_csv_data, stratified_split
from trainer import ModelTrainer, build_training_args
from model import build_model 

from tweet_logger import build_logger 
import time
import os



def main(config): 
    global tokenizer
    _ = set_seed()
    filename = os.path.basename(__file__)
    
    logger, file_handler, console_handler = build_logger(filename)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    
    logger.info("\n\nReading csv file and splitting data...")
    tweet_df = read_csv_data("data/tweets.csv", config.FEATURE_COL, config.LABEL_COL, encoding=config.ENCODING)
    
    try:
        train_df, test_df, val_df = stratified_split(tweet_df) #save the test data maybefor another script???
        
        logger.info(f"============ DATA SPLITS =============\
            \n=== val_df ===\n{val_df[config.LABEL_COL].value_counts()} \
                \n=== test_df ===\n{test_df[config.LABEL_COL].value_counts()} \
                     \n=== train_df ===\n{train_df[config.LABEL_COL].value_counts()}")
  
    except Exception as e: 
        logger.critical(f"A critical error occurred:\n\tNO CSV DATA RETREIVED")
        logger.exception(f"An exception occurred as {e}")
        logger.removeHandler(console_handler)
        logger.removeHandler(file_handler)
        return 
    
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
                                compute_metrics=compute_metrics)
    
    logger.debug("Building training arguments...") 
    training_args= build_training_args(logging_steps=25, max_steps=50)
    trainer = tuned_model.build_trainer(train_data, val_data, training_args)
    
    logger.info("============BATCH TRAINING INFORMATION============")
    logger.info(f"Train size: {len(train_data)}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Epochs: {config.NUM_EPOCHS}")
    logger.info(f"Steps per epoch: {len(train_data) // config.BATCH_SIZE}")
    logger.info(f"Approx total steps: {(len(train_data) // config.BATCH_SIZE) * config.NUM_EPOCHS}")
    
    logger.debug("Starting Training...")
    train_start = time.time()
    #trainer.train()
    train_stop = time.time()
    logger.debug("Training Complete") 
    train_time = train_stop - train_start
    logger.info(f"Total train time in seconds: {train_time}")

    logger.debug("Starting Evaluation...")   
    eval_start = time.time()
    #trainer.evaluate() 
    eval_stop = time.time()
    logger.debug("Evaluation Complete...")
    eval_time = eval_stop - eval_start 
    logger.info(f"Total evaluation time: {eval_time}")
    
    logger.info("Model training completed successfully with checkpoint saved")
    logger.removeHandler(console_handler)
    logger.removeHandler(file_handler)
    
    
if __name__ == "__main__":
    main(config)






