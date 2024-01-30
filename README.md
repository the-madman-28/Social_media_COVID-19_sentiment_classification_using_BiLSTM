# Social Media-Based COVID-19 Sentiment Classification Model Using Bi-LSTM

## Deep Learning Project

This project focuses on sentiment analysis of COVID-19-related tweets, utilizing a Bi-LSTM (Bidirectional Long Short-Term Memory) neural network. The model is designed to read tweets bidirectionally, providing a more comprehensive understanding of sentiment, whether it is Positive, Negative, or Neutral.

![COVID-19 Tweets](https://github.com/shivani1031/Social_media-based_COVID-19_sentiment_classification_model_using_Bi-LSTM/assets/69902161/cc0f087d-d933-4713-839e-575162479f59)

### Model Architecture

The utilized Bi-LSTM model is visualized below:

![Bi-LSTM Model](https://github.com/shivani1031/Social_media-based_COVID-19_sentiment_classification_model_using_Bi-LSTM/assets/69902161/605cddc0-f663-41ec-99c9-0f876d2ad392)

---

## Code Overview

### dl.py

The code performs the following tasks:

1. **Data Loading and Preprocessing:**
   - Loads the dataset from a CSV file (`Corona_NLP_train.csv`).
   - Cleans the data by converting text to lowercase, removing HTML tags, URLs, punctuation, stopwords, and emojis.

2. **Tokenization and Word Embeddings:**
   - Tokenizes the text using NLTK's sentence tokenizer.
   - Generates word embeddings using Word2Vec from the Gensim library.

3. **Label Encoding:**
   - Encodes sentiment labels using scikit-learn's LabelEncoder.

4. **Train-Test Split:**
   - Splits the dataset into training and testing sets using `train_test_split`.

5. **Model Building:**
   - Constructs a Bi-LSTM model with an embedding layer, Bidirectional LSTM layers, dropout layers, and a dense layer with softmax activation for multi-class sentiment classification.

6. **Model Training:**
   - Compiles and trains the model using the Adam optimizer and sparse categorical crossentropy loss.

7. **Model Evaluation:**
   - Plots training history (accuracy and loss) using Matplotlib.
   - Evaluates the model on the test set and prints accuracy and loss metrics.

8. **Prediction:**
   - Uses the trained model to make predictions on the test set.

### Important Note

- Ensure that the necessary libraries are installed, including NumPy, pandas, Keras, NLTK, Gensim, and TensorFlow.
- The dataset file `Corona_NLP_train.csv` should be available in the specified location.

Feel free to modify the code as needed and experiment with hyperparameters for improved results.

---

**Note:** The provided code and readme assume a Google Colab environment and Google Drive integration. Adjust paths and file access accordingly if running in a different environment.
