# Neural Machine Translation (English to Spanish) using Seq2Seq

## 🚀 Project Overview

This project is a Deep Learning model specializing in Natural Language Processing (NLP), built to translate sentences from English (source language) to Spanish (target language).

This model uses an advanced **Sequence-to-Sequence (Seq2Seq)** architecture, which consists of two main components: an **Encoder** and a **Decoder**. Both components were built using GRU (Gated Recurrent Unit) layers, an efficient type of Recurrent Neural Network (RNN).

---

## 📊 Dataset

* **Source:** A dataset from a `data.csv` file was used, containing a parallel corpus.
* **Content:** The file consists of two columns: `english` and `spanish`, where each row contains an English sentence and its corresponding Spanish translation.
* **Size:** The dataset contains 118,964 sentence pairs.

---
     
## ⚙️ Data Preprocessing

Text preprocessing is the most critical step to prepare the data for the model:

1.  **Text Cleaning:**
    * A `clean_text` function was created to clean all texts in both columns.
    * Cleaning operations included: converting all characters to lowercase, removing brackets, links, punctuation, and any words containing numbers.
2.  **Adding Start/End Tokens:**
    * A `<start>` token was added to the beginning of every sentence and an `<end>` token to its end.
    * These tokens are essential for the Seq2Seq model to know when to start and stop generating an answer.
3.  **Tokenization:**
    * `tf.keras.preprocessing.text.Tokenizer` was used to create a separate vocabulary for each language (one for English, one for Spanish).
    * Every text sentence was converted into a sequence of these numerical IDs.
4.  **Padding:**
    * Since sentences have different lengths, `tf.keras.preprocessing.sequence.pad_sequences` was used to make all sequences the same length by adding zero-padding to the end of shorter sentences.
5.  **Data Split:**
    * The entire dataset was split into 90% for training and 10% for testing.
6.  **Data Pipeline:**
    * To ensure efficient training, the data was converted into a `tf.data.Dataset`, which allows for shuffling and batching the data.

---

## 🤖 Model Architecture: Seq2Seq

The model relies on the **Encoder-Decoder** architecture:

### 1. The Encoder

The Encoder is a `tf.keras.Model` whose job is to "understand" the input English sentence.

* **Embedding Layer:** Converts the sequence of numbers into dense vectors.
* **GRU Layer:** Processes these vectors word-by-word, and returns the final "hidden state".
* **"Context Vector":** The final hidden state is a numerical summary, or "understanding," of the entire English sentence and is passed to the Decoder.

### 2. The Decoder

The Decoder is a `tf.keras.Model` whose job is to "generate" the Spanish sentence.

* **Initial State:** It begins by receiving the "context vector" from the Encoder as its initial hidden state.
* **GRU Layer:** Processes the input (starting with `<start>`) and the hidden state to generate the next word.
* **Dense Layer:** A final `Dense` layer with `softmax` activation, which predicts the next word in the sequence from the entire Spanish vocabulary.

---

## 📈 Training Process

* **Optimizer:** `Adam`.
* **Loss Function:** `SparseCategoricalCrossentropy`, using a mask to ignore padding (zeros) when calculating the loss.
* **Teacher Forcing:**
    * To speed up training, this technique was used. It involves feeding the Decoder the *correct* word from the target sentence at each step instead of the word it just predicted.
* **Model Saving:** The best Encoder and Decoder weights were saved based on the lowest loss achieved on the test data.

---

## 🔬 Inference & Translation

To translate a new sentence, the `translate` function works as follows:

1.  The input English sentence is cleaned, tokenized, and padded.
2.  It is passed through the **Encoder** to get the "context vector".
3.  The **Decoder** begins its process, starting with the `<start>` token and the context vector.
4.  The model enters a loop: it predicts the next word, then uses that prediction as the input for the following step (Autoregressive Decoding).
5.  This process continues until the model predicts the `<end>` token or reaches the maximum sentence length.

### Example Results

As shown in the screenshot, the model can successfully translate:

* `translate('how are you')` ➡️ `('how are you', 'cómo te vas')`
* `translate('book')` ➡️ `('book', 'libro')`

![Example Translations](MlCourse.ipynb%20-%20Visual%20Studio%20Code%2006_09_2024%2005_40_56%20م.png)

---

## 🛠️ Technologies Used

* Python 3
* TensorFlow & Keras
* Pandas
* Scikit-learn (for data splitting)
* NumPy
* re (for text cleaning)
