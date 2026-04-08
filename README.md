# Fine-Tuned Transformer Models for Tweet Sentiment Classification

## Overview

This project fine-tunes pretrained Hugging Face transformer models for tweet sentiment classification. Three models were compared on the same sampled Kaggle tweet dataset:

- DistilBERT
- BERT
- RoBERTa

The project extends earlier NLP work by moving from models built from scratch to transfer learning with pretrained language models.

Dataset link: [Kaggle Tweet Dataset](https://www.kaggle.com/datasets/bhavikjikadara/tweets-dataset/data)

Exploratory data analysis showed that most tweet lengths fell between 1 and 40 tokens, so a maximum sequence length of 40 was selected as a balance between efficiency and context retention.

Because the dataset is imbalanced, the data was split into training, validation, and test sets using stratification. Class weights were computed from the training data and used during model fine-tuning.

Among the three models, **RoBERTa** achieved the best overall performance and was selected for inference and API serving.

Best RoBERTa test results:

- **Accuracy:** 86%
- **F1-score:** 78%
- **ROC-AUC:** 93%

The project also includes:

- batch prediction on the test set
- inference for new text input
- a FastAPI app for local or cloud serving
- notebook-based error analysis

## Folder Structure

```text
transformer-model/
├── data/
├── logs/
├── notebook/
├── outputs/
│   ├── bert/
│   ├── distilbert/
│   └── roberta/
├── src/
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
├── README.md
└── requirements.txt
```
## Installation
Install the required packages:

```bash
pip install -r requirements.txt
```

## Training

Model settings are controlled in src/config.py through MODEL_KEY.

Available options:

- `distilbert`
- `bert`
- `roberta`

Training will:

- load the selected pretrained model and tokenizer
- prepare and split the dataset
- compute class weights
- fine-tune the model
- resume from checkpoint if available
- save the best model and tokenizer
- evaluate on the held-out test set
- save test metrics to test_metrics/ 

```bash
python src/train.py
```

Example saved model path: 
```bash
 outputs/roberta/best_model/
 ```

## Prediction / Inference

### Batch Prediction

Generates predictions for the test set and saves them to **data/.**

```bash
python src/predict.py
```
### Inference

Loads the trained model and predicts sentiment for a single string or a list of strings.

```bash
python src/inference.py
```

**Example output:**

```json
{'label': 'positive', 'score': 0.9315873980522156}
```

## API Serving

The FastAPI app loads the trained model once at startup and serves predictions through an API.

Run locally with:

```bash
uvicorn src.app:app --reload
```
## Notebooks 

The notebooks support the experimental workflow:

- **Tweet Data EDA.ipynb** — exploratory data analysis
- **Transformer-tuning.ipynb** — Colab fine-tuning workflow
- **Tweet Classification.ipynb** — evaluation and error analysis

The reusable pipeline is implemented in the src/ scripts.

## Limitations

- Only a sample of the original dataset was used
- Sequence length was capped at 40 tokens
- Threshold tuning was not explored
- Shared failure cases across all three models were not separately analysed

## Conclusion

This project presents an end-to-end transformer NLP workflow for tweet sentiment classification, covering data preparation, fine-tuning, evaluation, inference, and API serving.

It also compares DistilBERT, BERT, and RoBERTa on the same task and highlights the trade-off between efficiency and predictive performance. RoBERTa achieved the best overall results and was selected as the final model.