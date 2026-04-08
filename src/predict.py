import config 
from data import read_csv_data
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

    
def predict(texts, model, tokenizer,  device, batch_size=config.BATCH_SIZE, max_length=config.MAX_SEQ_LEN):
        all_prediction = []
        all_probability = []
        texts = texts.tolist() if hasattr(texts ,"tolist") else list(texts)

        model.to(device)
        model.eval()
        for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                inputs = tokenizer(
                        batch_texts,
                        padding = True,
                        truncation=True,
                        return_tensors="pt", 
                        max_length=max_length
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
        
        model_path = config.BEST_MODEL_PATH
        
        tweet_test_df = read_csv_data(
                config.TEST_DATA_PATH, 
                config.FEATURE_COL, 
                 config.LABEL_COL, 
                 encoding=config.ENCODING
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        texts  = tweet_test_df[config.FEATURE_COL].tolist()
        
        prediction, probability = predict(texts, model, tokenizer, device)
        
        tweet_test_df["prediction"] = prediction
        tweet_test_df["probability"] = probability 
        tweet_test_df.to_csv(config.PREDICTION_DF_NAME,  index=False)
        
        print("Prediction complete")