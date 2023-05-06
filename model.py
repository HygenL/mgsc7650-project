import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.pipeline import Pipeline
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Load Data
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
data1 = pd.read_csv('all_kindle_review .csv')
data1 = data1.sample(n=3000, random_state=42)
data3 = pd.read_csv('kindle_reviews.csv')

# Define conversion function
def convert_to_binary(value):
    if value >3:
        return 1
    elif value <=3:
        return 0
    else:
        return value

data1['rating'] = data1['rating'].apply(convert_to_binary)

# Load nlp
nlp = spacy.load("en_core_web_sm")

# Define text preprocessing function
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.isalpha()]
    preprocessed_text = " ".join(tokens)
    return preprocessed_text

# Preprocess data
data1['combined'] = data1['reviewText'] + ' ' + data1['summary'].apply(preprocess_text)

# Split data
X = data1['combined']
y = data1['rating']
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# Set up tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Tokenize train and test data
train_encodings = tokenizer(X_train.to_list(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(x_test.to_list(), truncation=True, padding=True, max_length=512)

# Create dataset class
class KindleReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create train and test datasets
train_dataset = KindleReviewsDataset(train_encodings, Y_train)
test_dataset = KindleReviewsDataset(test_encodings, y_test)

# Set up training arguments and trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train model
trainer.train()

# Evaluate
# Evaluate model
eval_results = trainer.evaluate()

# Make predictions
predictions = trainer.predict(test_dataset)
predicted_class = predictions.predictions.argmax(-1)

# Calculate accuracy and F1 score
test_accuracy = accuracy_score(y_test, predicted_class)
test_f1 = f1_score(y_test, predicted_class)

# Print results
print("Test Accuracy: ", test_accuracy)
print("Test F1 Score: ", test_f1)

# Save model and tokenizer
model.save_pretrained("BERT")
tokenizer.save_pretrained("BERT")

# Load model and tokenizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

loaded_tokenizer = DistilBertTokenizer.from_pretrained("BERT")
loaded_model = DistilBertForSequenceClassification.from_pretrained("BERT")

# Create a new trainer with the loaded model
loaded_trainer = Trainer(
    model=loaded_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Evaluate
eval_results = loaded_trainer.evaluate()

# Make predictions
predictions = loaded_trainer.predict(test_dataset)
predicted_class = predictions.predictions.argmax(-1)

# Calculate accuracy and F1 score
test_accuracy = accuracy_score(y_test, predicted_class)
test_f1 = f1_score(y_test, predicted_class)

# Print results
print("Test Accuracy: ", test_accuracy)
print("Test F1 Score: ", test_f1)

