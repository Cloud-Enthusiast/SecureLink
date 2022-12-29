from sklearn.feature_extraction import CountVectorizer
import gensim
import numpy as np
import pandas as pd

# Load the word2vec model
model = gensim.models.Word2Vec.load("path/to/word2vec/model")

# Define a function to convert text to a vector


def text_to_vector(text):
    # Tokenize the text
    words = text.split()

    # Get the embedding for each word
    word_vectors = [model[word] for word in words if word in model]

    # Return the average of the word vectors
    return np.mean(word_vectors, axis=0)


# Example text
text = "This is an example sentence."

# Convert the text to a vector
vector = text_to_vector(text)
