import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import (
    InputLayer,
    TimeDistributed,
    SpatialDropout1D,
    Bidirectional,
)
from tensorflow import keras
from sklearn.model_selection import train_test_split
import json
import pickle


def sentence_integrate(data):
    agg_func = lambda s: [
        (w, p, t)
        for w, p, t in zip(
            s["Word"].values.tolist(),
            s["POS"].values.tolist(),
            s["Tag"].values.tolist(),
        )
    ]
    return data.groupby("Sentence").apply(agg_func).tolist()


max_len = 50
df_total = pd.read_csv("dataset/ner_dataset_tratado.csv", encoding="latin1")
df_total = df_total.fillna(method="ffill")

sentences = sentence_integrate(df_total)

words = list(set(df_total["Word"].values))
words.append("ENDPAD")
print(words[:10])
tags = list(set(df_total["Tag"].values))
num_tags = len(tags)

phrases = [" ".join([w[0] for w in s]) for s in sentences]

# Tokenizer para as frases
tokenizer = Tokenizer(oov_token="UNK")
tokenizer.fit_on_texts(phrases)

# Convert sentences to sequences of integer tokens
X = tokenizer.texts_to_sequences(phrases)

# Salvar o Tokenizer em um arquivo
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=0)

tag2idx = {t: i for i, t in enumerate(tags)}
y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(
    maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"]
)

with open("data/tag2idx.json", "w") as fp:
    json.dump(tag2idx, fp)

print("exportado")

# Update num_words and the input size of the Embedding layer to be the size of
#  the tokenizer's vocabulary
num_words = (
    len(tokenizer.word_index) + 1
)  # added 1 to account for padding token 0


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

model = keras.Sequential()
model.add(InputLayer((max_len)))
model.add(
    Embedding(input_dim=num_words, output_dim=max_len, input_length=max_len)
)
model.add(SpatialDropout1D(0.1))
model.add(
    Bidirectional(
        LSTM(units=100, return_sequences=True, recurrent_dropout=0.1)
    )
)

# compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test, y_test),
    batch_size=32,
    epochs=5,
    verbose=1,
)

model.save("data/model.h5")

print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss: {} ".format(results[0]))
print("test accuracy: {} ".format(results[1]))
