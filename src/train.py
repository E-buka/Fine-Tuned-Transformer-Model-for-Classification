from transformers import DataCollatorWithPadding
import config 
from seed import set_seed
from utils import tokenizer 
from data import MakeDataset, read_csv_data, stratified_split
from trainer import ModelTrainer, build_training_args
from model import Models 
from utils import compute_metrics
import time

def main(config): 
    global tokenizer
    seed = set_seed()
    
    #log 
    tweet_df = read_csv_data("data/tweets.csv", config.FEATURE_COL, config.LABEL_COL, encoding=config.ENCODING)

    # log 
    train_df, test_df, val_df = stratified_split(tweet_df)
    #log 
    print(val_df[config.LABEL_COL].value_counts())
    print(test_df[config.LABEL_COL].value_counts())
    print(train_df[config.LABEL_COL].value_counts())
    
    #log 
    tokenizer = tokenizer(config.MODEL_TOKENIZER)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    #log 
    train_data = MakeDataset(train_df[config.FEATURE_COL], train_df[config.LABEL_COL], tokenizer)
    test_data = MakeDataset(test_df[config.FEATURE_COL], test_df[config.LABEL_COL], tokenizer)
    val_data = MakeDataset(val_df[config.FEATURE_COL], val_df[config.LABEL_COL], tokenizer)
    
    #log 
    select_model = Models(num_labels=2)
    model = select_model.distilBERT() 

    distilbert_model= ModelTrainer(model=model,
                                tokenizer=tokenizer,
                                data_collator=data_collator,
                                compute_metrics=compute_metrics)
    
    # log 
    training_args= build_training_args(logging_steps=25, max_steps=50)
    trainer = distilbert_model.build_trainer(train_data, val_data, training_args)
    
    ## print to log 
    print("Train size:", len(train_data))
    print("Batch size:", config.BATCH_SIZE)
    print("Epochs:", config.NUM_EPOCHS)
    print("Steps per epoch:", len(train_data) // config.BATCH_SIZE)
    print("Approx total steps:", (len(train_data) // config.BATCH_SIZE) * config.NUM_EPOCHS)
    
    train_start = time.time()
    #trainer.train()
    train_stop = time.time()
    # log 
    train_time = train_stop - train_start
    print(train_time)

    # log    
    eval_start = time.time()
    trainer.evaluate() 
    eval_stop = time.time()
    eval_time = eval_stop - eval_start 
    print(eval_time)
    
    # log all done successfully
    
    
if __name__ == "__main__":
    main(config)






