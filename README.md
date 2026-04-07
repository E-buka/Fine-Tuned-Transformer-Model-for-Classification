# Tuned Transformer Model: Tweet Sentiment Classifier 

## Project Summary

This project uses huggingface transformer models for tweet sentiment classification.  It uses a subset of the kaggle tweet dataset to fine tune three transformer models: distilBERT, BERT and RoBERTa. The project also builds from a previous NLP-foundation project where a deep learning model was built from scratch. 
data link: [kaggle] (https://www.kaggle.com/datasets/bhavikjikadara/tweets-dataset/data)

The EDA for the subsampled dataset is contained in a notebook. The EDA revealed that the sequence length for the dataset was between 1 and 40. Therefore, a maximum sequence length of 40 was selected in the tokenization for speed and performance trade-off. 

The sampled tweet dataset was split with stratification due to class imbalance into training, validation and testing dataset. The testing dataset was saved and set aside for testing the models in the end and the class weights were computed from the training dataset. For each model, a pretrained tokenizer and model was loaded and fine-tuned with the tweet dataset. The best model (roberta) was saved for prediction and inference after achieving accuracy of 86%, F1 score - 78% and ROC-AUC score - 93%. The models performance metrics are saved as json files. An offline inference and api serving was created from the best performing model. 

The predictive performance for the three models were evaluated on the test data and an error analysis was conducted on the predicted data. This is contained in a notebook. The error analysis revealed that tweets with sequence length of 15 or less were predicted with higher accuracy and confidence than tweets with sequence length of 16 or more. 

## Folder Structure 

```text
transformer-model/
├── data/
│   └── roberta_pred_df.csv
│   └── sample_tweet.csv
├── logs/
├── notebook/
│   ├── Transformer-tuning.ipynb
│   ├── Tweet Classification.ipynb
│   └── Tweet Data EDA.ipynb
├── outputs/
│   ├── bert/
│   ├── distilbert/
│   └── roberta/
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── data.py
│   ├── inference.py
│   ├── model.py
│   ├── predict.py
│   ├── seed.py
│   ├── train.py
│   ├── trainer.py
│   ├── tweet_logger.py
│   └── utils.py
├── test_metrics/
├── .gitignore
├── README.md
└── requirements.txt
```
## Installation
create and activate a virtual environment, then install requirments

```bash
pip install -r requirements.txt
```

## Training

Training parameters are contained in `config.py` file. Three models are available and can be selected by choosing a model key from the list of keys: 
- `distilbert`
- `bert`
- `roberta`

Then run the `train.py` file. The training process loads the selected pretrained model and corresponding pretrained tokenizer. It acquires a  logging handler and creates a logfile, loads the dataset, splits and saves the test dataset then computes class weight from the train data. It the trains the model and save the checkpoints to the output directory. The model resumes from the last checkpoint if the training is paused or terminated unexpectedly. Once training is complete the best model is saved to `outputs/model_name/best_model.pt`. The model performs a final test of the test data and saves the outcome to the test_metrics directory.  

```bash
python train.py
```

## Prediction / Inference

The predict.py file loads the saved best model and tokenizer through  predict() and predicts the test data in batches. The predicted labels adn probabilities are added as columns to the dataset and saved as csv file in data directory. 

The inference script accepts list of strings or a string and outputs the predicted label and score

```bash
python predict.py
```

```bash
python inference.py
```

## Sample prediction output
```json
{'label': 'positive', 'score': 0.9315873980522156}
```

## api serving

The `app.py` uses FastAPI for online serving.  It loads the pretrained model and get user input through the POST method and returns a json result of predicted label and score . 

It can be hosted locally with uvicorn.
```bash
uvicorn app:app
```

## Limitation
1. The project uses only a subsample of the tweet dataset for fine tuning the models and with a maximum sequence lenght of 40.  This meant that the tokenizer can potentiall truncate useful contexts due to the small sequence length.  
2. The error analysis did not evaluate conditions where the three models all predicted wrong class for a tweet. This could give further insight to the model performance on dataset features. 
3. The tuned models did not optimise classification threshold for the tweet which could improve the accuracy of the models. 
