import pandas as pd
from keras.models import load_model
import json
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_bag_of_words():
    with open("data/tag2idx.json", "r") as fp:
        tag2idx = json.load(fp)

    with open("data/word2idx.json", "r") as fp:
        word2idx = json.load(fp)

    num_words = max(word2idx.values())

    return word2idx, tag2idx, num_words


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


df = pd.read_csv("dataset/ner_dataset_test.csv")
df = df.fillna(method="ffill")

model = load_model("data/model.h5")
word2idx, tag2idx, num_words = create_bag_of_words()

print("______________________________________")
print("OBAMA ->", word2idx["Obama"])
print("______________________________________")


sentences = sentence_integrate(df)

max_len = 50

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(
    maxlen=max_len, sequences=X, padding="post", value=num_words - 1
)

y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(
    maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"]
)

results = model.evaluate(X, y, batch_size=128)
print("test loss, test acc:", results)
print("test loss: {} ".format(results[0]))
print("test accuracy: {} ".format(results[1]))
