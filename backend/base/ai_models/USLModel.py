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

#from googletrans import Translator
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['figure.figsize'] = (10, 10
                                  )
plt.rcParams['axes.grid'] = False
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8
colors = ['xkcd:pale range', 'xkcd:sea blue', 'xkcd:pale red', 'xkcd:sage green', 'xkcd:terra cotta', 'xkcd:dull purple', 'xkcd:teal', 'xkcd: goldenrod', 'xkcd:cadet blue',
          'xkcd:scarlet']
bbox_props = dict(boxstyle="round,pad=0.3", fc=colors[0], alpha=.5)


true_data = pd.read_csv('D:/Users/Shekhar/SecureLink/Dataset/True.csv')

false_data = pd.read_csv('D:/Users/Shekhar/SecureLink/Dataset/Fake.csv')

true_data.head()

false_data.head()

true_data['Target'] = ['True']*len(true_data)

false_data['Target'] = ['False']*len(false_data)

data = true_data.append(false_data).sample(
    frac=1).reset_index().drop(columns=['index'])

data['label'] = pd.get_dummies(data.Target)['False']

# Train-Test Validation

train_text, temp_text, train_labels, temp_labels = train_test_split(data['title'], data['label'],
random_state=2018,
stratify=data['Target'])
test_size = 0.3,
