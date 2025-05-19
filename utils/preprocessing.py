import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Classification hyperparams
CLASS_LABELS   = ["Math","Science","History","English"]
MAX_VOCAB_CLASS= 5000
CLASS_MAXLEN   = 20

# Generation hyperparams
GEN_VOCAB_SIZE = 10000
GEN_SEQ_LEN    = 10

def preprocess_classification(texts, labels):
    tok = Tokenizer(num_words=MAX_VOCAB_CLASS, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    seqs = tok.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=CLASS_MAXLEN, padding="post")
    label2idx = {l:i for i,l in enumerate(CLASS_LABELS)}
    y = to_categorical([label2idx[l] for l in labels], num_classes=len(CLASS_LABELS))
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42), tok

def preprocess_generation(corpus):
    words = corpus.lower().split()
    if len(words) <= GEN_SEQ_LEN:
        raise ValueError("Corpus too small.")
    tok = Tokenizer(num_words=GEN_VOCAB_SIZE, oov_token="<OOV>")
    tok.fit_on_texts(words)
    vocab = min(GEN_VOCAB_SIZE, len(tok.word_index)+1)
    windows = [words[i:i+GEN_SEQ_LEN+1] for i in range(len(words)-GEN_SEQ_LEN)]
    seqs = [tok.texts_to_sequences([" ".join(w)])[0] for w in windows]
    seqs = np.array([s for s in seqs if len(s)==GEN_SEQ_LEN+1])
    X, y = seqs[:,:GEN_SEQ_LEN], seqs[:,GEN_SEQ_LEN]
    y = to_categorical(y, num_classes=vocab)
    return (X, y), tok, vocab
