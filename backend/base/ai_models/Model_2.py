import os
import re
from tkinter import Y
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow_hub as hub
from sklearn.decomposition import PCA
from transformers import AutoModel, BertTokenizerFast
import transformers
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

true_news = pd.read_csv('D:/Users/Shekhar/SecureLink/Dataset/True.csv')

fake_news = pd.read_csv('D:/Users/Shekhar/SecureLink/Dataset/Fake.csv')

true_news.head()

fake_news.head()

ps = WordNetLemmatizer()
stopwords = stopwords.words('english')
nltk.download('wordnet')

def cleaning_data(row):
    row = row.lower()
    row = re.sub('[^a-zA-z]', ' ', row)
    token = row.split()
    news = [ps.lemmatize(word) for word in token if not word in stopwords]
    cleaned_news = ''.join(news)
    return cleaned_news

train_data, test_data, train_label, test_label = train_test_split(X,y,test_size=0.2, random_state=0)