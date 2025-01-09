from flask import Flask, request, jsonify
import joblib
import re
from nltk.stem.porter import PorterStemmer
import nltk
import os

nltk.download('stopwords')

# Initialize the Flask application
app = Flask(__name__)

# Absolute paths for the model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.joblib')
MODEL_PATH = os.path.join(BASE_DIR, 'model.joblib')

# Load the pre-trained model and vectorizer
try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()

port_stem = PorterStemmer()

def stemming(content):
    if not isinstance(content, str):
        content = str(content)
    stemmed_content = re.sub(r'\W+', ' ', content)
    stemmed_content = ' '.join([port_stem.stem(word) for word in stemmed_content.split()])
    return stemmed_content

# Define a route for the default URL, which loads the form
@app.route('/')
def form():
    return '''
        <form action="/predict" method="post">
            <textarea name="news_content" rows="10" cols="30"></textarea>
            <br>
            <input type="submit" value="Classify">
        </form>
    '''

# Define a route for the predict URL, which handles the form submission and returns a prediction
@app.route('/predict', methods=['POST'])
def predict():
    news_content = request.form['news_content']
    stemmed_input = stemming(news_content)
    vectorized_input = vectorizer.transform([stemmed_input])
    prediction = model.predict(vectorized_input)
    result = "The news is Real" if prediction[0] == 0 else "The news is Fake"
    return jsonify({'result': result})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
