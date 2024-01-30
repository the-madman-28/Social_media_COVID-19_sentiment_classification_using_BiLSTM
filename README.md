# Social_media-based_COVID-19_sentiment_classification_model_using_Bi-LSTM
## Deep learning Project

The dataset consist of the tweets made by the people during the covid-19 period. My model is trained using Bi-LSTM, as it can read the tweet from front as well as back. And therefore, can let us know whether the sentiment is Positive, Negative or neutral.

<img width="630" alt="image" src="https://github.com/shivani1031/Social_media-based_COVID-19_sentiment_classification_model_using_Bi-LSTM/assets/69902161/cc0f087d-d933-4713-839e-575162479f59">
<br> <br>
<h3>The model used is : </h3>
<img width="370" alt="image" src="https://github.com/shivani1031/Social_media-based_COVID-19_sentiment_classification_model_using_Bi-LSTM/assets/69902161/605cddc0-f663-41ec-99c9-0f876d2ad392">

# Sentiment Analysis using BiLSTM and Word Embeddings

## Overview

This repository implements a sentiment analysis model using a Bidirectional Long Short-Term Memory (BiLSTM) neural network with word embeddings. The dataset used for training and testing is the "Corona_NLP_train.csv" dataset, which contains tweets related to the COVID-19 pandemic.

## Setup and Requirements

1. **Dataset:**
   - The dataset used is located in the "archive" directory and is named "Corona_NLP_train.csv".
   - Ensure that the dataset is correctly formatted and available in the specified path.

2. **Libraries:**
   - The code requires several Python libraries, including NumPy, pandas, Keras, NLTK, Gensim, and Matplotlib. Make sure these libraries are installed in your environment.

3. **Google Drive Integration:**
   - The code assumes that it is running in a Google Colab environment and uses Google Drive for file access.
   - If running locally or in a different environment, adjust file paths accordingly.

## Code Execution

1. **Data Preprocessing:**
   - The code begins with loading the dataset and performing various data cleaning steps, including converting text to lowercase, removing HTML tags, URLs, punctuation, stopwords, and emojis.

2. **Tokenization and Word Embeddings:**
   - The text is tokenized using NLTK's sentence tokenizer, and word embeddings are generated using Word2Vec from the Gensim library.

3. **Label Encoding:**
   - Sentiment labels are encoded using scikit-learn's LabelEncoder.

4. **Train-Test Split:**
   - The dataset is split into training and testing sets using scikit-learn's `train_test_split` function.

5. **Model Architecture:**
   - The model architecture consists of an embedding layer, Bidirectional LSTM layers with dropout, and a dense layer with softmax activation for multi-class sentiment classification.

6. **Model Training:**
   - The model is compiled using the Adam optimizer and sparse categorical crossentropy loss. Training is performed for a specified number of epochs and batch size.

7. **Model Evaluation:**
   - The code includes plots for training history (accuracy and loss) and evaluates the model on the test set, providing accuracy and loss metrics.

8. **Prediction:**
   - The trained model is used to make predictions on the test set, and the results are printed and visualized.

## Important Note

- Ensure that you have the required permissions to access files in the specified directories.

Feel free to modify the code as needed and experiment with hyperparameters to achieve better results.
