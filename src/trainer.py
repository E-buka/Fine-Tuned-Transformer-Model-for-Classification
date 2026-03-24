from transformers import TrainingArguments, Trainer 
import config 

def build_training_args(**kwargs):
    return TrainingArguments(
                output_dir = config.OUTPUT_DIR , 
                learning_rate = config.LEARNING_RATE, 
                per_device_train_batch_size = config.BATCH_SIZE, 
                per_device_eval_batch_size = config.BATCH_SIZE,
                num_train_epochs = config.NUM_EPOCHS, 
                weight_decay = config.WEIGHT_DECAY, 
                save_strategy = "epoch", 
                push_to_hub = False,
                logging_dir = config.LOG_DIR,
                **kwargs
            )
    
    
class ModelTrainer:
    def __init__(self, model, tokenizer, data_collator, compute_metrics):
        
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
           
    def build_trainer(self, train_data, eval_data, train_args):
        return Trainer(
            model = self.model, 
            args = train_args,
            train_dataset = train_data, 
            eval_dataset = eval_data, 
            processing_class= self.tokenizer,
            data_collator = self.data_collator, 
            compute_metrics = self.compute_metrics 
            
        )


