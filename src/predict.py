import config 
from data import read_csv_data
from seed import set_seed
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



model_path = config.BEST_MODEL_PATH

_ = set_seed()

tweet_test_df = read_csv_data(
        config.TEST_DATA_PATH, 
        config.FEATURE_COL, 
        config.LABEL_COL, 
        encoding=config.ENCODING
        )


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

texts  = tweet_test_df[config.FEATURE_COL].tolist()

    
def predict(texts, model, tokenizer, batch_size=config.BATCH_SIZE, max_length=config.MAX_SEQ_LEN):
        all_prediction = []
        all_probability = []
        texts = texts.tolist() if hasattr(texts ,"tolist") else list(texts)

        for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                inputs = tokenizer(
                        batch_texts,
                                   padding = True,
                                   truncation=True,
                                   return_tensors="pt", 
                                   max_lenght=max_length
                                   )
                
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1)
                        pred = torch.argmax(probs, dim=1)
                
                all_prediction.extend(pred.cpu().tolist())
                all_probability.extend(probs.cpu().tolist())
        return all_prediction, all_probability 

if __name__ == "__main__":
        prediction, probability = predict(texts, model, tokenizer)
        tweet_test_df["prediction"] = prediction
        tweet_test_df["probability"] = probability 
        tweet_test_df.to_csv(config.PREDICTION_DF_NAME)
        print("Prediction complete")