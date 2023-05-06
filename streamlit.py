# download data from firestore
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.pipeline import Pipeline
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import streamlit as st
from google.cloud import firestore
import os
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.svm import LinearSVC
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

# # Preprocess data
data1['combined'] = data1['reviewText'] + ' ' + data1['summary'].apply(preprocess_text)

# # Split data
X = data1['combined']
y = data1['rating']
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# # Set up tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# # Tokenize train and test data
train_encodings = tokenizer(X_train.to_list(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(x_test.to_list(), truncation=True, padding=True, max_length=512)

# # Create dataset class
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

# # Set up training arguments and trainer
training_args = TrainingArguments(
     output_dir='./results',
     num_train_epochs=3,
     per_device_train_batch_size=8,
     per_device_eval_batch_size=8,
     logging_dir='./logs',
 )

# # Load model and tokenizer
loaded_tokenizer = DistilBertTokenizer.from_pretrained("BERT")
loaded_model = DistilBertForSequenceClassification.from_pretrained("BERT")
loaded_trainer = Trainer(model=loaded_model, args=training_args,train_dataset=train_dataset,eval_dataset=test_dataset,
 )
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "spring-yifan.json"
db = firestore.Client()
# getting data from your firestore database - reddit collection
reddit = db.collection(u'book-reddit')
posts = list(reddit.stream())
docs_dict = list(map(lambda x: x.to_dict(), posts))
df = pd.DataFrame(docs_dict)
df.to_csv("output_dataset.csv", index=False)
df = df.dropna(subset=['created'])
data2 = df.loc[:, ['selftext', 'sentiment']]
df_test = df.loc[:, ['selftext', 'sentiment']]

# Fill missing or NaN values with an empty string
data2['selftext'] = data2['selftext'].fillna('')

# Encode the data2 dataset
data2['combined'] = data2['selftext'].apply(preprocess_text)
data2_encodings = tokenizer(data2['combined'].tolist(), truncation=True, padding=True, max_length=512)

# Create data2 class
class Data2Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# convert the predictions back to the original labels
def convert_from_binary(value):
    if value == 1:
        return 'positive'
    elif value == 0:
        return 'negative'
    else:
        return value

    #Loading Spinner
with st.spinner('Wait for it...'):
    time.sleep(5)
st.success('Page Loaded!')

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "spring-yifan.json"

st.title("Sentiment Analyzer")

home, model, team = st.tabs(["Home", "Model", "Team"])

with home:
    st.subheader('Explanation of Project')
    st.write('For this project, we wanted to create a model that would predict the sentiment of books reviews. We used a file found on Kaggle to train our model, and then we implemented an NLP model on reviews from Reddit. With Google Cloud, we were able to get reviews in real time saved to a FireStore database.')
    st.write('Here is a wordcloud from our FireStore database.')
    st.image('wordcloud.png', width = 500, use_column_width= True)
    st.divider()
    st.subheader('A simple model was done through Google Cloud. Here are the sentiment predictions from our database.')
    counts = data2.groupby('sentiment').size().reset_index(name='count')
    fig, ax = plt.subplots()
    colors = ['darkblue', 'darkorange']
    ax.bar(counts['sentiment'], counts['count'],color = colors)
    # Set the chart title and axis labels
    ax.set_title('Sentiment Counts')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    # getting data from your firestore database - reddit collection
    st.divider()
    created_end = datetime.fromtimestamp(df.iloc[:1,:].created.values[0])
    created_start = datetime.fromtimestamp(df.iloc[-1:,:].created.values[0])
    if created_start > created_end:
        created_start, created_end = created_end, created_start

    date_start = st.sidebar.date_input("From", value=created_start, min_value=created_start, max_value=created_end)
    date_end = st.sidebar.date_input("To", value=created_end, min_value=created_start, max_value=created_end)
    posts_length_range = st.sidebar.slider("Posts Length", min_value=1, max_value=9999, value=[1, 9999])

    date_start_str = date_start.strftime('%Y-%m-%d')
    date_end_str = date_end.strftime('%Y-%m-%d')
    df['date'] = df['created'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
    df = df.loc[(df.date >= date_start_str) & (df.date <= date_end_str), :]
    
    df['length'] = df['selftext'].apply(lambda x: len(x))
    df = df.loc[(df.length >= posts_length_range[0]) & (df.length <= posts_length_range[1]), :]
    
    chart = st.columns([2,1])
    fig = px.histogram(df, x='date', color='sentiment', color_discrete_map={"positive": "brown", "negative": "darkgreen"}, barmode="group")
    st.plotly_chart(fig)
    st.caption("sentiment on subreddit r/RomanceBooks")


    "---"

    st.subheader("Sample Posts")
    placeholder = st.empty()
    with placeholder.container():
        for index, row in df.sample(10).iterrows():
            text = row["selftext"].strip()
            if text != "":
                col1, col2 = st.columns([3,1])
                with col1:
                    with st.expander(text[:100] + "..."):
                        st.write(text)
                with col2:
                    if row["sentiment"] == "positive":
                        st.info(row['sentiment'])
                    else:
                        st.error(row['sentiment'])

#-----------------------------------------------------------------------------------------

with model:
    st.subheader("Model Training Dataset")
    st.dataframe(data3.sample(5))
    st.divider() #-------------------
    st.subheader("Our LinearSVC Model")
    col1, col2, col3 = st.columns(3)
    with col1:
        col1.metric("Accuracy", "81%", "1%")
    with col2:
        col2.metric("F1 Score", "0.82", "0.02")
    with col3:
        st.write("We used LinearSVC model first in the Google cloud function and achieved a relatively high accuracy and F1 score, \
            but we want to explore other models with better performance")
    st.divider() #------------------
    st.subheader("Our DistilBert Model")
    col1, col2, col3 = st.columns(3)
    with col1:
        col1.metric("Accuracy", "89%", "1.2%")
    with col2:
        col2.metric("F1 Score", "0.90", "0.006")
    with col3:
        st.write("According to our results of testing different kinds of models, DistilBert Model performed best")

    # st.text("a pre-trained DistilBERT model with a classification head added")
    # st.text("And then we tokenized the datasets using the tokenizer.")
    # st.text("Create a custom PyTorch dataset class (KindleReviewsDataset) to store the tokenized data and corresponding labels.")
    # st.text("Create instances of the custom dataset class for the training and test datasets.")
    # st.text("Set up training arguments and a Trainer instance from the Hugging Face Transformers library.")
    # st.text("Train the model using the Trainer instance.")

# define prediction function
    def predict_sentiment(input_text):
        preprocessed_text = preprocess_text(input_text)
        input_encoding = tokenizer([preprocessed_text], truncation=True, padding=True, max_length=512)
        input_dataset = Data2Dataset(input_encoding)
        input_prediction = loaded_trainer.predict(input_dataset)
        predicted_class = input_prediction.predictions.argmax(-1)
        return convert_from_binary(predicted_class[0])

    st.subheader("User Interface")
    selftext = st.text_input("Enter review here")

    if st.button("Get Prediction"):
    # Get the model's prediction
        prediction = predict_sentiment(selftext)

    # Display the prediction
        st.write(f"The model's prediction is: {prediction}")
        if prediction == "positive":
            st.write("The result is: Positive")
        else:
            st.write("The result is: Negative")
    
    #--------
with team:
    st.subheader("About")
    "We are a group of five students from Tulane"
    "We all are pusuing a Master's in Business Analytics."
    "We hope you like our application :smile:"
    st.subheader("Team")
    "---"
    col1, col2 = st.columns([1,3])
    with col1:
        st.image("member.png")
    with col2:
        st.subheader("Member Names")
        st.write("🔵 Mahrukh Khattak 🇵🇰")
        st.write("💗 Haizhen Liu 🇨🇳")
        st.write("🟠 Yifan Wang 🇨🇳")
        st.write("💚 Jack Spencer 🇺🇸")
        st.write("🟣 Will Steinhorn 🇺🇸")