from transformers import pipeline 

# do i need this
#trainer.push_to_hub()

model_link = ROOT/"models"
sentiment_model = pipeline(model=model_link)

example_texts = ["try this!", "and that"] 

sentiment_model(example_texts)

