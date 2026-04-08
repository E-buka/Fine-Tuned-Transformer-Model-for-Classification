from transformers import pipeline 
import config 
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_model():
    
    model_path = config.BEST_MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.config.id2label = {0: "negative", 1: "positive"}
    model.config.label2id = {"negative": 0, "positive": 1}
    model.eval()

    classifier = pipeline(task='text-classification', 
                    model=model, 
                    tokenizer=tokenizer
                    )
    
    return classifier



def tweet_predictor(text: str, pipeline):
    output = pipeline(text)
    return output[0] 

