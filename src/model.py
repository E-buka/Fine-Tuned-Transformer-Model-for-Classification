from transformers import AutoModelForSequenceClassification 

class Models:
    def __init__(self, num_labels):
        self.num_labels = num_labels
        
    def distilbert(self):
        return AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=self.num_labels)

    def bert(self):
        return AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=self.num_labels)

    def roberta(self):
        return AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=self.num_labels)
    

def build_model(model_name, num_labels):
    model = Models(num_labels)
    if model_name == "distilbert":
        return model.distilbert()
    elif model_name == "bert":
        return model.bert()
    elif model_name == "roberta":
        return model.roberta()
    else:
        raise ValueError("Enter a model name")
        