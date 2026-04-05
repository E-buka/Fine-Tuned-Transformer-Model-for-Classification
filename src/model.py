from transformers import AutoModelForSequenceClassification 

class Models:
    def __init__(self, num_labels):
        self.num_labels = num_labels
        
    def distilBERT(self):
        return AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=self.num_labels)

    def BERT(self):
        return AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=self.num_labels)

    def RoBERTa(self):
        return AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=self.num_labels)
    

def build_model(model_name, num_labels):
    model = Models(num_labels)
    if model_name == "distilBERT":
        return model.distilBERT()
    elif model_name == "BERT":
        return model.BERT()
    elif model_name == "RoBERTa":
        return model.RoBERTa()
    else:
        raise ValueError("Enter a model name")
        