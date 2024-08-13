import streamlit as st
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = load_files('path/to/your/dataset')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

# Create a TfidfVectorizer object
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the training data and transform both the training and testing data
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Create a MultinomialNB object
classifier = MultinomialNB()

# Train the classifier using the training data
classifier.fit(X_train_vectors, y_train)

# Create a Streamlit app
st.title('Sentiment Analysis App')

# Text input
text = st.text_area('Enter text to analyze')

# Analyze button
if st.button('Analyze'):
    # Transform the text into a vector
    vector = vectorizer.transform([text])
    
    # Predict the sentiment
    sentiment = classifier.predict(vector)[0]
    
    # Display the sentiment
    st.write(f'Sentiment: {sentiment}')
