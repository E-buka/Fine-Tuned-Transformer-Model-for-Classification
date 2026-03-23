import numpy as np
import evaluate
from transformers import AutoTokenizer 



def compute_metrics(eval_pred):
    
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")
    load_recall = evaluate.load("recall")
    load_precision = evaluate.load("precision")
    
    logits, label = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = load_accuracy.compute(predictions=predictions, references=label)
    f1 = load_f1.compute(predictions=predictions, references=label)
    precision = load_precision.compute(predictions=predictions, references=label)
    recall = load_recall.compute(predictions=predictions, references=label)
    
    return {"accuracy": accuracy, 
            "f1_score": f1, 
            "precision":precision, 
            "recall":recall}
    
    

def tokenizer(model_tokenizer):
    return AutoTokenizer.from_pretrained(model_tokenizer)

