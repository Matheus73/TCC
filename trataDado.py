import pandas as pd


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


df = pd.read_csv("dataset/ner_dataset.csv", encoding="latin1")
df = df.fillna(method="ffill")

df.rename(columns={"Sentence #": "Sentence"}, inplace=True)

df.to_csv("dataset/ner_dataset_tratado.csv", index=False)

sentences = sentence_integrate(df)

print("O valor total de sentenças é: ", len(sentences))
div_size = int(len(sentences) * 0.5)
# divide the dataset in 20% for training and 80% for testing
twenty_percent = int(len(sentences) * 0.2)
print("O valor de 20% das sentenças é: ", twenty_percent)
print("O valor de 50% das sentenças é: ", div_size)

div_index = df.query(f"Sentence == 'Sentence: {div_size}'").tail(1).index[0]

df_train = df.iloc[:div_index + 1]
df_unlabeled = df.iloc[div_index + 1:]

df_train.to_csv("dataset/ner_dataset_train.csv", index=False)
df_unlabeled.to_csv("dataset/ner_dataset_unlabeled.csv", index=False)
# 524314