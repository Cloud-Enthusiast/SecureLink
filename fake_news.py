import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("D:/securelink/SecureLink-main/SecureLink-main/Dataset/fake-news/train.csv")
df.dropna(axis=0, inplace = True)
df["combined"] = df["author"] + " " + df["title"]
print(df.head())

nltk.download("stopwords")

def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words("english")]

message_bow = CountVectorizer(analyzer=process_text).fit_transform(df["combined"])
X_train, X_test, y_train, y_test = train_test_split(message_bow, df["label"], test_size = 0.2, random_state = 0)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)
print(pred)
print(y_test)
