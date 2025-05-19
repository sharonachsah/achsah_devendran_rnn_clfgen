import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dropout, Dense, Layer
)
import tensorflow.keras.backend as K
from utils.data_utils import build_classification_dataset
from utils.preprocessing import preprocess_classification

# custom attention
class Attention(Layer):
    def __init__(self, **kw): 
        super().__init__(**kw)
        self.score = Dense(1)
    def call(self, x):
        s = K.squeeze(self.score(x), -1)
        a = K.softmax(s)
        a = K.expand_dims(a, -1)
        return K.sum(x * a, axis=1)

def build_model(vocab_size, maxlen, embed_dim=50, units=64, n_classes=4):
    inp = Input(shape=(maxlen,))
    x = Embedding(vocab_size, embed_dim)(inp)
    x = Bidirectional(LSTM(units, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(units, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Attention()(x)
    out = Dense(n_classes, activation="softmax")(x)
    m = Model(inp, out)
    m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return m

def main():
    texts, labels = build_classification_dataset()
    (X_tr, X_te, y_tr, y_te), tok = preprocess_classification(texts, labels)
    vocab_size = min(len(tok.word_index)+1, 5000)
    model = build_model(vocab_size, X_tr.shape[1])
    model.fit(X_tr, y_tr, epochs=40, batch_size=32, validation_split=0.1)
    print("Eval:", model.evaluate(X_te, y_te))
    model.save("datasets/rnn_classifier.keras")

if __name__=="__main__":
    main()
