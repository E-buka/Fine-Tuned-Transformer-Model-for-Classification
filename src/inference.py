from transformers import pipeline 
import config 
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

model_path = config.BEST_MODEL_PATH
device = Accelerator().device

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.config.id2label = {0: "negative", 1: "positive"}
model.config.label2id = {"negative": 0, "positive": 1}
model.eval()



def tweet_predictor(text: str, device=device, pipeline=pipeline, model=model, tokenizer=tokenizer):
    
    pipeline = pipeline(task='text-classification', 
                        model=model, 
                        tokenizer=tokenizer,
                        device=device)
    output = pipeline(text)
    return json.dumps(output[0]) 

if __name__ == "__main__":
    print(tweet_predictor("oh hahaha, "))