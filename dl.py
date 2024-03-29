
!cd /content/drive/MyDrive/dlproj

!unzip /content/drive/MyDrive/dlproj/archive.zip

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from numpy import array

from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPool1D, Embedding, Conv1D, LSTM
from sklearn.model_selection import train_test_split

#importing the imdb dataset

import pandas as pd

# sentiment_review= pd.read_csv('/content/Corona_NLP_train.csv')
df = pd.read_csv('/content/archive/Corona_NLP_train.csv', encoding='MacRoman')

df.shape

df.head(5)

df.Sentiment.value_counts()

df['OriginalTweet'][3]

"""Data Cleaning"""

df['OriginalTweet'][3].lower()

df['OriginalTweet'] = df['OriginalTweet'].str.lower()

df

#remove tags, html
import re

def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)

import pandas as pd

# Assuming you have a DataFrame called df with a column 'OriginalTweet'
# Replace this with your actual DataFrame

# Example:
# df = pd.read_csv('your_data.csv')

df['OriginalTweet'] = df['OriginalTweet'].apply(remove_html_tags)

df['OriginalTweet']

#removing url
def remove_url(text):
  pattern = re.compile(r'https?://\S+|www\.\S+')
  return pattern.sub(r'', text)

df['OriginalTweet'] = df['OriginalTweet'].apply(remove_url)

import string
string.punctuation

exclude = string.punctuation

def remove_pun(text):
  for char in exclude:
    text = text.replace(char, ' ')
  return text

df['OriginalTweet'] = df['OriginalTweet'].apply(remove_pun)

df['OriginalTweet']

def remove_punc1(text):
  return text.translate(str.maketrans(' ', ' ', exclude))

df['OriginalTweet']

import nltk
from nltk.corpus import stopwords

# Download the NLTK stopwords corpus
nltk.download('stopwords')

# Now you can access the English stopwords
stop_words = set(stopwords.words('english'))

# Define your remove_stopwords function
def remove_stopwords(text):
    words = text.split()  # Split the text into words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Apply the remove_stopwords function to the 'OriginalTweet' column
df['OriginalTweet'] = df['OriginalTweet'].apply(remove_stopwords)

#to remove the emoji
import re

def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (10s)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

df['OriginalTweet'] = df['OriginalTweet'].apply(remove_emoji)

"""#Tokenization"""

import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd

# Assuming you have a DataFrame called df with a column 'OriginalTweet'
# Replace this with your actual DataFrame

# Example:
# df = pd.read_csv('your_data.csv')

# Download the NLTK punkt tokenizer
nltk.download('punkt')

# Apply sentence tokenization to each element in the 'OriginalTweet' column
df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: sent_tokenize(str(x)))

df['OriginalTweet']

from tensorflow.keras.preprocessing.text import Tokenizer

# Assuming 'OriginalTweet' is the column containing preprocessed text in your training data
texts = df['OriginalTweet'].astype(str).tolist()

# Create a tokenizer and fit on the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Get the vocabulary size
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 for the padding token

print("Vocabulary Size:", vocab_size)

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd



# Train Word2Vec model
model = Word2Vec(sentences=df['OriginalTweet'], vector_size=100, window=5, min_count=1, workers=4)

# Function to convert a tweet to a vector
def tweet_to_vector(tweet):
    vector = []
    for word in tweet:
        if word in model.wv:
            vector.append(model.wv[word])
    return vector

# Apply the function to each row in the DataFrame
df['OriginalTweet'] = df['OriginalTweet'].apply(tweet_to_vector)

df['OriginalTweet']

df['OriginalTweet'].ndim

df.head(5)

from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Fit and transform the 'sentiment' column
df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

# Display the mapping between original labels and encoded labels
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

df['Sentiment']

(df['OriginalTweet'][3000])

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Assuming X_train and X_test are your padded sequences with shape (num_samples, max_seq_length, embedding_dim)
# Ensure that your input sequences have the correct shape
max_seq_length =128
X_train = pad_sequences(train_df['OriginalTweet'].tolist(), maxlen=max_seq_length, padding='post', truncating='post', dtype='float32')
y_train = train_df['Sentiment']

X_test = pad_sequences(test_df['OriginalTweet'].tolist(), maxlen=max_seq_length, padding='post', truncating='post', dtype='float32')
y_test = test_df['Sentiment']

# Check the shape of the sequences
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print(y_train.shape)
print("Shape of X_train:", X_train.shape, y_train.shape)
X_train=X_train[:,:,-1]
X_test=X_test[:,:,-1]
# y_train =y_train[]
print("Shape of X_train:", X_train.shape, y_train.shape)
print("Shape of X_test:", X_test.shape, y_test.shape)

# y_test_encoded = label_encoder.transform(y_test)

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the BiLSTM model with dropout layers
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=150, input_length=max_seq_length))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.5))  # Add a dropout layer with a dropout rate of 0.5
model.add(Bidirectional(LSTM(32, return_sequences=False)))
model.add(Dropout(0.5))  # Add another dropout layer with a dropout rate of 0.5
model.add(Dense(3, activation='softmax'))  # Assuming 3 classes (positive, negative, neutral)

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

import matplotlib.pyplot as plt
# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss, accuracy = model.evaluate(X_test, y_test)
print(accuracy)
print(loss)

print(X_test.shape)
print(X_test[0])

Y_pred = model.predict(X_test)

print(Y_pred.shape)
print(Y_pred[0])

from google.colab import drive
drive.mount('/content/drive')

Y_pred_labels = [np.argmax(i) for i in Y_pred]
print(Y_pred_labels)

