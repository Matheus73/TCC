import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import json
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.layers import (
    InputLayer,
    TimeDistributed,
    SpatialDropout1D,
    Bidirectional,
)
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences


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


# -----------------------------------------------------
# CONSTRUINDO O VOCABULÁRIO
# -----------------------------------------------------

df_total = pd.read_csv("dataset/ner_dataset_tratado.csv", encoding="latin1")
df_total = df_total.fillna(method="ffill")


words = list(set(df_total["Word"].values))
words.append("ENDPAD")
num_words = len(words)
print("num_words ---------- ", num_words)
tags = list(set(df_total["Tag"].values))
num_tags = len(tags)

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}


with open("data/word2idx.json", "w") as fp:
    json.dump(word2idx, fp)

with open("data/tag2idx.json", "w") as fp:
    json.dump(tag2idx, fp)


print("______________________________________")
print("OBAMA ->", word2idx["Obama"])
print("______________________________________")

# -----------------------------------------------------
# PREPARANDO OS DADOS
# -----------------------------------------------------

data = pd.read_csv("dataset/ner_dataset_train.csv")
data = data.fillna(method="ffill")

sentences = sentence_integrate(data)
max_len = 50

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(
    maxlen=max_len, sequences=X, padding="post", value=num_words - 1
)
print(X[0])

y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(
    maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"]
)


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# -----------------------------------------------------
# CONSTRUINDO O MODELO
# -----------------------------------------------------


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

# -----------------------------------------------------
# TESTANDO O MODELO
# -----------------------------------------------------

print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss: {} ".format(results[0]))
print("test accuracy: {} ".format(results[1]))
