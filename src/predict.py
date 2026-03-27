from transformers import pipeline 



sentiment_model = pipeline(model='')

example_texts = ["try this!", "and that"] 

sentiment_model(example_texts)

