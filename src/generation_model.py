from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Embedding, GRU, RepeatVector,
    Flatten, Concatenate, Dense, Attention as KerasAttn
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.data_utils import build_generation_corpus
from utils.preprocessing import preprocess_generation

GEN_SEQ_LEN = 10

def build_model(vocab_size, seq_len=GEN_SEQ_LEN, embed_dim=50, units=128):
    enc_in = Input((seq_len,))
    e = Embedding(vocab_size, embed_dim)(enc_in)
    enc_out, enc_state = GRU(units, return_state=True, return_sequences=True)(e)
    dec_in = RepeatVector(1)(enc_state)
    dec_out = GRU(units, return_sequences=True)(dec_in, initial_state=enc_state)
    ctx = KerasAttn()([dec_out, enc_out])
    ctx = Flatten()(ctx); d = Flatten()(dec_out)
    concat = Concatenate()([ctx, d])
    out = Dense(vocab_size, activation="softmax")(concat)
    m = Model(enc_in, out)
    m.compile(loss="categorical_crossentropy", optimizer="adam")
    return m

def generate(model, tok, seed, n_words=20):
    result = seed.split()
    for _ in range(n_words):
        seq = pad_sequences([ [tok.word_index.get(w,0) for w in result[-GEN_SEQ_LEN:]] ],
                             maxlen=GEN_SEQ_LEN)
        p = model.predict(seq)[0]
        result.append(tok.index_word.get(p.argmax(), "<OOV>"))
    return " ".join(result)

def main():
    corpus = build_generation_corpus()
    (X, y), tok, vocab = preprocess_generation(corpus)
    model = build_model(vocab)
    model.fit(X, y, epochs=50, batch_size=64)
    model.save("datasets/rnn_generator.keras")
    seed = "alice was beginning to get very tired of"
    print("Generated:", generate(model, tok, seed))

if __name__=="__main__":
    main()
