from transformers import AutoModelForSequenceClassification 

class Models:
    def __init__(self, num_labels):
        self.num_labels = num_labels
        
    def distilBERT(self):
        return AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=self.num_labels)

    
    def BERT(self):
        return AutoModelForSequenceClassification.from_pretrained("BERT", num_labels=self.num_labels)

    
    def aROBERTo(self):
        return AutoModelForSequenceClassification("aROBERTo", num_labels=self.num_labels)