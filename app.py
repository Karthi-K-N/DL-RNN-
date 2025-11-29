import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# load the IMDb dataset word index

word_index=imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

#load model
model = load_model('rnnmodel.keras')

#step 2: helper function
# funtion to decode reviews

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review=[word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


#step 3: prediction function

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Step 4: steamlit app
import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

#user input
user_input = st.text_area('Movie review')

if st.button('Classify'):
    sentiment, prediction = predict_sentiment(user_input)
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction score" {prediction}')
else:

    st.write('please enter movie review.')
