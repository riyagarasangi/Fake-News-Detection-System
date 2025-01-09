import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import joblib

nltk.download('stopwords')

# Load the dataset
news_dataset = pd.read_csv('F:\\2nd yearminorproject\\datasets\\train.csv')

# Preprocess the data
port_stem = PorterStemmer()

def stemming(content):
    if not isinstance(content, str):
        content = str(content)
    stemmed_content = re.sub(r'\W+', ' ', content)
    stemmed_content = ' '.join([port_stem.stem(word) for word in stemmed_content.split()])
    return stemmed_content

news_dataset['content'] = news_dataset['author'].fillna('') + ' ' + news_dataset['title'].fillna('') + ' ' + news_dataset['text'].fillna('')
news_dataset['content'] = news_dataset['content'].apply(stemming)

X = news_dataset['content'].values
Y = news_dataset['label'].values

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)

# Train a logistic regression model
model = LogisticRegression(max_iter=200)  # Increased max_iter for better convergence
model.fit(vectorizer.transform(X_train), Y_train)

# Save the vectorizer and the model to disk
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(model, 'model.joblib')

print("Model and vectorizer saved.")
