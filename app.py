#!/usr/bin/env python3
"""
simple_rnn_nlp.py

A self-contained script that:
 1. Downloads and caches:
    - A small educational text classification dataset (Math, Science, History, English)
    - A longer corpus for next-word generation (Project Gutenberg excerpt)
 2. Preprocesses data (tokenization, sequencing, padding, one-hot labels)
 3. Builds and trains:
    - A SimpleRNN-based classifier
    - A SimpleRNN-based language model for next-word prediction
 4. Evaluates classification accuracy
 5. Demonstrates next-word generation

Usage:
    python simple_rnn_nlp.py
"""

import os
import sys
import requests
import time
import urllib.request
import zipfile
import numpy as np
import random
import warnings
from pathlib import Path
from tensorflow.keras.layers import (
    Input,
    Embedding,
    GRU,
    RepeatVector,
    TimeDistributed,
    Attention as KerasAttention,
    Flatten,
    Concatenate,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    GRU,
    Dense,
    Dropout,
    Input,
    Layer,
)
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import GRU, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)
CLASS_DATA_FILE = CACHE_DIR / "edu_class_data.txt"
GEN_DATA_FILE = CACHE_DIR / "gutenberg_corpus.txt"
CLASS_MODEL_FILE = CACHE_DIR / "rnn_classifier.keras"
GEN_MODEL_FILE = CACHE_DIR / "rnn_generator.keras"

# Classification settings
CLASS_LABELS = ["Math", "Science", "History", "English"]
MAX_VOCAB_CLASS = 5000
CLASS_MAXLEN = 20
CLASS_EMBED_DIM = 50
CLASS_RNN_UNITS = 64
CLASS_BATCH = 32
CLASS_EPOCHS = 40

# Generation settings
GEN_VOCAB_SIZE = 10000
GEN_SEQ_LEN = 10  # input sequence length
GEN_EMBED_DIM = 50
GEN_RNN_UNITS = 128
GEN_BATCH = 64
GEN_EPOCHS = 50

# ------------------------------------------------------------------------------
# UTILITIES: Download & Cache
# ------------------------------------------------------------------------------


def download_and_cache(url, filepath, unzip=False):
    """Download a file from URL if not present; optionally unzip."""
    if filepath.exists():
        print(f"[cache] {filepath} already exists, skipping download.")
        return
    print(f"[download] Fetching {url} ...")
    tmpfile, _ = urllib.request.urlretrieve(url)
    if unzip:
        print(f"[unzip] Extracting to {CACHE_DIR} ...")
        with zipfile.ZipFile(tmpfile, "r") as z:
            z.extractall(CACHE_DIR)
    else:
        os.replace(tmpfile, filepath)
    print(f"[ok] Saved to {filepath}.")


# ------------------------------------------------------------------------------
# PHASE 1: DATASET CREATION
# ------------------------------------------------------------------------------


def build_classification_dataset():
    """
    Fetches up to 200 introductory paragraphs per class label from Wikipedia
    category pages, caches them, and returns parallel lists (texts, labels).
    """
    # If cached, just load
    if CLASS_DATA_FILE.exists():
        print("[load] Loading classification dataset from cache.")
        lines = CLASS_DATA_FILE.read_text(encoding="utf-8").splitlines()
    else:
        print("[build] Downloading classification dataset from Wikipedia...")
        # Map your labels to Wikipedia category titles
        LABEL_CATS = {
            "Math": "Category:Mathematics",
            "Science": "Category:Science",
            "History": "Category:History",
            "English": "Category:English_language",
        }
        lines = []
        session = requests.Session()
        API = "https://en.wikipedia.org/w/api.php"

        for label, cat in LABEL_CATS.items():
            # 1) Get up to 200 pages in that category
            cm_params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": cat,
                "cmlimit": 200,
                "format": "json",
            }
            resp = session.get(API, params=cm_params).json()
            members = resp.get("query", {}).get("categorymembers", [])

            # collect pageids
            pageids = [str(m["pageid"]) for m in members]
            # 2) batch by 20 to get extracts
            for i in range(0, len(pageids), 20):
                batch = pageids[i : i + 20]
                ex_params = {
                    "action": "query",
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "pageids": "|".join(batch),
                    "format": "json",
                }
                ex_resp = session.get(API, params=ex_params).json()
                pages = ex_resp.get("query", {}).get("pages", {})
                for pid, page in pages.items():
                    text = page.get("extract", "").strip().replace("\n", " ")
                    # skip very short or missing intros
                    if len(text) < 50:
                        continue
                    # add one example
                    lines.append(f"{label}\t{text}")
                # be kind to the API
                time.sleep(0.1)

        # shuffle and cache
        random.shuffle(lines)
        CLASS_DATA_FILE.write_text("\n".join(lines), encoding="utf-8")
        print(f"[cache] Saved {len(lines)} samples to {CLASS_DATA_FILE!r}.")

    # Finally parse into two lists
    texts, labels = [], []
    for ln in lines:
        lab, txt = ln.split("\t", 1)
        labels.append(lab)
        texts.append(txt)
    return texts, labels


def build_generation_corpus():
    """
    Downloads a Project Gutenberg plain text (e.g., 'Alice in Wonderland'),
    caches it, and returns its contents as a single string.
    """
    if GEN_DATA_FILE.exists():
        print("[load] Loading generation corpus from cache.")
        return GEN_DATA_FILE.read_text(encoding="utf-8")
    # Example: Alice's Adventures in Wonderland by Lewis Carroll
    gutenberg_url = "https://www.gutenberg.org/files/11/11-0.txt"
    print("[download] Fetching Gutenberg corpus...")
    try:
        raw = (
            urllib.request.urlopen(gutenberg_url)
            .read()
            .decode("utf-8", errors="ignore")
        )
    except Exception as e:
        print(f"[error] Failed to download: {e}")
        sys.exit(1)

    # Strip header/footer
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK 11 ***"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK 11 ***"
    start = raw.find(start_marker)
    end = raw.find(end_marker)

    if start != -1 and end != -1 and end > start:
        # skip past the start marker line
        start = raw.find("\n", start) + 1
        corpus = raw[start:end]
    else:
        # fallback to entire text if markers aren’t found
        corpus = raw

    print("[cache] Saving corpus...")
    GEN_DATA_FILE.write_text(corpus, encoding="utf-8")
    return corpus


# ------------------------------------------------------------------------------
# PHASE 2: PREPROCESSING
# ------------------------------------------------------------------------------


def preprocess_classification(texts, labels):
    # Tokenize & build vocab
    tok = Tokenizer(num_words=MAX_VOCAB_CLASS, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    sequences = tok.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences, maxlen=CLASS_MAXLEN, padding="post", truncating="post"
    )
    # Encode labels
    label2idx = {lab: i for i, lab in enumerate(CLASS_LABELS)}
    y = np.array([label2idx[l] for l in labels])
    y = to_categorical(y, num_classes=len(CLASS_LABELS))
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        padded, y, test_size=0.2, stratify=y, random_state=42
    )
    return tok, X_train, X_test, y_train, y_test


def preprocess_generation(corpus):
    # 1. Tokenize into words
    words = corpus.lower().split()
    if len(words) <= GEN_SEQ_LEN:
        raise ValueError(
            f"Corpus too small ({len(words)} words); need > {GEN_SEQ_LEN}."
        )

    # 2. Fit tokenizer on the full corpus
    tok = Tokenizer(num_words=GEN_VOCAB_SIZE, oov_token="<OOV>")
    tok.fit_on_texts(words)
    actual_vocab = min(GEN_VOCAB_SIZE, len(tok.word_index) + 1)

    # 3. Build sliding windows
    seqs = [words[i : i + GEN_SEQ_LEN + 1] for i in range(len(words) - GEN_SEQ_LEN)]
    sentences = [" ".join(window) for window in seqs]

    # 4. Convert windows to integer sequences
    seqs_int = tok.texts_to_sequences(sentences)
    seqs_int = [s for s in seqs_int if len(s) == GEN_SEQ_LEN + 1]
    seqs_int = np.array(seqs_int)

    # 5. Split into X and y
    X = seqs_int[:, :GEN_SEQ_LEN]
    y = seqs_int[:, GEN_SEQ_LEN]

    # 6. One-hot encode y using actual_vocab
    y = to_categorical(y, num_classes=actual_vocab)

    return tok, X, y, actual_vocab


# ------------------------------------------------------------------------------
# PHASE 3: MODEL BUILDING
# ------------------------------------------------------------------------------


# Simple Bahdanau-style attention layer


@tf.keras.utils.register_keras_serializable(package="Custom", name="Attention")
class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = Dense(1, name="attn_score_dense")

    def call(self, inputs):
        scores = self.score_dense(inputs)  # (batch, timesteps, 1)
        scores = K.squeeze(scores, axis=-1)  # (batch, timesteps)
        alphas = K.softmax(scores, axis=1)  # (batch, timesteps)
        alphas = K.expand_dims(alphas, axis=-1)  # (batch, timesteps, 1)
        context = K.sum(inputs * alphas, axis=1)  # (batch, features)
        return context

    def get_config(self):
        config = super().get_config()
        return config


def build_classification_model(vocab_size):
    inp = Input(shape=(CLASS_MAXLEN,), name="text_input")
    x = Embedding(input_dim=vocab_size, output_dim=CLASS_EMBED_DIM, name="embed")(inp)

    # 1st Bidirectional LSTM
    x = Bidirectional(LSTM(CLASS_RNN_UNITS, return_sequences=True), name="bilstm_1")(x)
    x = Dropout(0.3, name="drop_1")(x)

    # 2nd Bidirectional LSTM
    x = Bidirectional(LSTM(CLASS_RNN_UNITS, return_sequences=True), name="bilstm_2")(x)
    x = Dropout(0.3, name="drop_2")(x)

    # Attention pooling
    x = Attention(name="attention")(x)  # now returns (batch, features)

    # Final dense
    out = Dense(len(CLASS_LABELS), activation="softmax", name="classifier_output")(x)

    model = Model(inputs=inp, outputs=out, name="bi_lstm_attn_classifier")
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy", "Precision", "Recall"],
    )
    return model


def build_generation_model(vocab_size):
    # 1) Encoder
    encoder_inputs = Input(shape=(GEN_SEQ_LEN,), name="enc_input")
    enc_emb = Embedding(
        input_dim=vocab_size, output_dim=GEN_EMBED_DIM, name="enc_embed"
    )(encoder_inputs)
    # return_sequences=True to supply the full sequence to attention
    enc_outputs, enc_state = GRU(
        GEN_RNN_UNITS, return_sequences=True, return_state=True, name="encoder_gru"
    )(enc_emb)

    # 2) Decoder (single‐step for next‐word prediction)
    #    We feed the encoder state as initial state
    #    and use a dummy time dimension of 1 for the decoder input.
    decoder_inputs = RepeatVector(1, name="dec_repeat")(enc_state)
    dec_outputs = GRU(GEN_RNN_UNITS, return_sequences=True, name="decoder_gru")(
        decoder_inputs, initial_state=enc_state
    )

    # 3) Attention: query = decoder outputs, value = encoder outputs
    #    KerasAttention will compute dot(query, key) internally.
    #    It returns a context sequence of length 1.
    context_seq = KerasAttention(name="attention")(
        [dec_outputs, enc_outputs]
    )  # shape: (batch, 1, units)

    # 4) Flatten both context and decoder outputs, then concatenate
    context = Flatten(name="attn_flatten")(context_seq)  # (batch, units)
    dec_feat = Flatten(name="dec_flatten")(dec_outputs)  # (batch, units)
    concat = Concatenate(name="dec_concat")([context, dec_feat])

    # 5) Final projection to vocabulary
    outputs = Dense(vocab_size, activation="softmax", name="dec_output")(concat)

    model = Model(inputs=encoder_inputs, outputs=outputs, name="seq2seq_gru_attn")
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


# ------------------------------------------------------------------------------
# PHASE 4: TRAINING & EVALUATION
# ------------------------------------------------------------------------------


def train_and_eval_classifier(tok, X_train, X_test, y_train, y_test):
    vocab_size = min(MAX_VOCAB_CLASS, len(tok.word_index) + 1)
    if CLASS_MODEL_FILE.exists():
        print("[load] Loading saved classifier model.")
        model = load_model(CLASS_MODEL_FILE, custom_objects={"Attention": Attention})
    else:
        model = build_classification_model(vocab_size)
        print(model.summary())
        model.fit(
            X_train,
            y_train,
            validation_split=0.1,
            batch_size=CLASS_BATCH,
            epochs=CLASS_EPOCHS,
        )
        model.save(CLASS_MODEL_FILE)
    results = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    # print(
    #     f"Loss: {results['loss']:.4f}, Accuracy: {results['accuracy']:.4f}, "
    #     f"Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}"
    # )
    print(results)

    # Demonstrate on 5 random test samples
    idxs = np.random.choice(len(X_test), 5, replace=False)
    for i in idxs:
        pred = model.predict(X_test[i : i + 1])
        true_lab = CLASS_LABELS[np.argmax(y_test[i])]
        pred_lab = CLASS_LABELS[np.argmax(pred)]
        txt = " ".join(tok.index_word.get(idx, "") for idx in X_test[i] if idx != 0)
        print(f"  » Text: {txt!r}\n    True: {true_lab}, Pred: {pred_lab}")
    return model


def generate_text(model, tok, seed_text, n_words=20):
    result = seed_text.split()
    for _ in range(n_words):
        seq = tok.texts_to_sequences([result[-GEN_SEQ_LEN:]])
        seq = pad_sequences(seq, maxlen=GEN_SEQ_LEN, padding="pre")
        pred = model.predict(seq, verbose=0)
        idx = np.argmax(pred)
        word = tok.index_word.get(idx, "<OOV>")
        result.append(word)
    return " ".join(result)


def train_and_demo_generator(tok, X, y, actual_vocab):
    # Build or load the model with the same vocab size
    if GEN_MODEL_FILE.exists():
        print("[load] Loading saved generator model.")
        model = load_model(GEN_MODEL_FILE)
    else:
        model = build_generation_model(actual_vocab)
        print(model.summary())
        model.fit(X, y, batch_size=GEN_BATCH, epochs=GEN_EPOCHS)
        model.save(GEN_MODEL_FILE)

    # Demo text generation
    seed = "alice was beginning to get very tired of".lower()
    print(f"\n[demo] Seed text: {seed!r}")
    gen = generate_text(model, tok, seed, n_words=20)
    print(f"[demo] Generated: {gen}")
    return model


# ------------------------------------------------------------------------------
# MAIN ROUTINE
# ------------------------------------------------------------------------------


def main():
    # 1. Build or load datasets
    texts, labels = build_classification_dataset()
    corpus = build_generation_corpus()

    # 2. Preprocess
    class_tok, X_tr, X_te, y_tr, y_te = preprocess_classification(texts, labels)
    gen_tok, X_gen, y_gen, gen_vocab = preprocess_generation(corpus)

    # 3. Train & evaluate classifier
    print("\n=== Training & Evaluating Classifier ===")
    clf_model = train_and_eval_classifier(class_tok, X_tr, X_te, y_tr, y_te)

    # 4. Train & demo generator
    print("\n=== Training & Demonstrating Generator ===")
    gen_model = train_and_demo_generator(gen_tok, X_gen, y_gen, gen_vocab)


if __name__ == "__main__":
    main()
