from transformers import TrainingArguments, Trainer 
import torch.nn as nn
import config 
from transformers import EarlyStoppingCallback 



def build_training_args(**kwargs):
    return TrainingArguments(
                output_dir = config.OUTPUT_DIR , 
                learning_rate = config.LEARNING_RATE, 
                lr_scheduler_type = config.LR_TYPE,
                per_device_train_batch_size = config.BATCH_SIZE, 
                per_device_eval_batch_size = config.BATCH_SIZE,
                num_train_epochs = config.NUM_EPOCHS, 
                weight_decay = config.WEIGHT_DECAY, 
                eval_strategy= "epoch",
                save_strategy = "epoch", 
                push_to_hub = False,
                logging_dir = config.LOG_DIR,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                max_grad_norm=1.0,
                restore_callback_states_from_checkpoint = True,
                **kwargs
            )
    
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights 
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits 
        
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


    
class ModelTrainer:
    def __init__(self, model, tokenizer, data_collator, compute_metrics, class_weights=None):
        
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.class_weights = class_weights
           
    def build_trainer(self, train_data, val_data, training_args):
        return WeightedTrainer(
            model = self.model, 
            args = training_args,
            train_dataset = train_data, 
            eval_dataset = val_data, 
            processing_class= self.tokenizer,
            data_collator = self.data_collator, 
            compute_metrics = self.compute_metrics,
            class_weights=self.class_weights, 
            callbacks = [EarlyStoppingCallback(early_stopping_patience= 3, 
                                               early_stopping_threshold=0.001),]
        )


