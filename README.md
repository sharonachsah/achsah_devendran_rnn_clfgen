

# 🧠 Educational Text Classifier & Generator

This project is a dual-purpose NLP system that:

1. **Classifies** short educational texts into categories like **Math**, **Science**, **History**, and **English** using a Bidirectional LSTM with attention.
2. **Generates** text in the style of classic literature using a GRU-based sequence-to-sequence model with attention.

It leverages TensorFlow/Keras, Wikipedia for classification data, and Project Gutenberg for generation data.

---

## 📦 Features

* **Text Classification:**

  * Fetches introductory Wikipedia articles by category.
  * Preprocesses and tokenizes text for RNN training.
  * Uses a BiLSTM + Attention model to classify into predefined subjects.

* **Text Generation:**

  * Downloads *Alice in Wonderland* from Project Gutenberg.
  * Trains a GRU-based sequence-to-sequence model with attention for next-word prediction.
  * Supports seed-based generation.

---

## 🧰 Dependencies

This project requires Python 3.7+ and the following libraries:
```bash
py -3.11 -m venv venv
venv\Scripts\activate
```bash

Install requirements with:

```bash
pip install tensorflow numpy requests scikit-learn
```

Python 3.7+ recommended.

---

## 📁 Project Structure

```
.
├── your_script.py
├── data_cache/
│   ├── edu_class_data.txt         # Cached Wikipedia data
│   ├── gutenberg_corpus.txt       # Cached Gutenberg text
│   ├── rnn_classifier.keras       # Trained classifier model
│   └── rnn_generator.keras        # Trained generator model
```

---

## 🚀 How to Run

Run the main script:

```bash
python app.py
```

It will:

1. Download and cache datasets.
2. Train the classifier and evaluate on test samples.
3. Train the generator and produce sample output from a seed.

---

## 🧪 Output Examples

### Classification

```text
Text: 'mathematics is the study of numbers and...'
True: Math, Predicted: Math
```

### Text Generation

```text
Seed: "alice was beginning to get very tired of"
Generated: "alice was beginning to get very tired of sitting by her sister on the bank and of having nothing"
```

---

## 🧠 Models

### Classifier: BiLSTM + Attention

* Bidirectional LSTMs extract rich features.
* Attention mechanism pools time-distributed representations.

### Generator: GRU Seq2Seq + Attention

* Encoder-decoder GRUs with attention on the encoder outputs.
* Generates one word at a time based on previous sequence.

---

## ⚙️ Configuration

Edit these parameters at the top of the script:

```python
CLASS_LABELS = ["Math", "Science", "History", "English"]
CLASS_EPOCHS = 40
GEN_EPOCHS = 50
GEN_SEQ_LEN = 10
```

---

## 📚 Data Sources

* **Wikipedia API** for classification data.
* **[Project Gutenberg](https://www.gutenberg.org/)** for generation corpus.

---

## 📄 License

This project uses public domain datasets and is provided as-is for educational use.

---
