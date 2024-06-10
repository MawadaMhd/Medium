pip install transformers
#Loading BERT and its Tokenizer: We'll load the BERT model and its tokenizer, which transforms text into a format that BERT can understand:

from transformers import BertTokenizer, BertForSequenceClassification
import torch
# Load the BERT-base-uncased model  
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2) # 2 labels for positive/negative

# Preprocessing Text: We'll feed sample movie reviews to the tokenizer, transforming them into a format that BERT can process:

# Sample movie reviews
text = ["This movie was amazing!", "I really disliked this book."]
# Tokenize and encode the text
encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

#Fine-Tuning: We'll fine-tune BERT on a dataset of movie reviews, teaching it to distinguish between positive and negative sentiments.

# Load dataset and split into training and validation sets
# … (load dataset and split)
# Define the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()
# Train the model
# … (train loop)
# Evaluate the model on the validation set
# … (evaluation loop)

#Making Predictions:Let's see if BERT can correctly classify new movie reviews:

# Get predictions for new text
new_text = ["This product is fantastic!"]
encoded_input = tokenizer(new_text, padding=True, truncation=True, return_tensors='pt')
outputs = model(**encoded_input)
predicted_class = torch.argmax(outputs.logits, dim=1)
print(predicted_class)
