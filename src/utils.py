import numpy as np
import evaluate
import torch
from transformers import AutoTokenizer 
from sklearn.metrics import average_precision_score, roc_auc_score

load_accuracy = evaluate.load("accuracy")
load_f1 = evaluate.load("f1")
load_recall = evaluate.load("recall")
load_precision = evaluate.load("precision")

def compute_metrics(eval_pred):   
    global average_precision_score
    global roc_auc_score
    logits, label = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    logits = torch.from_numpy(logits)
    probs = torch.softmax(logits, dim=-1)
    y_score = probs[:, 1]
    
    accuracy = load_accuracy.compute(predictions=predictions, references=label)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=label)["f1"]
    precision = load_precision.compute(predictions=predictions, references=label)["precision"]
    recall = load_recall.compute(predictions=predictions, references=label)["recall"]
    average_precision = average_precision_score(label, y_score)
    roc_auc = roc_auc_score(label, y_score)
    
    return {"accuracy": accuracy, 
            "f1": f1, 
            "precision":precision, 
            "recall":recall,
            "average_precision": average_precision,
            "roc_auc": roc_auc,
            }
    
    

def tokenizer(model_tokenizer):
    return AutoTokenizer.from_pretrained(model_tokenizer)

