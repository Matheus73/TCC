import json
import os
import pickle
import time
from typing import Any, TypeAlias

# import nltk
import pandas as pd
import requests
import schedule

# from modAL.batch import uncertainty_batch_sampling
# from modAL.models.learners import ActiveLearner
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.sparse._csr import csr_matrix

# from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from keras.models import load_model

# from xgboost import XGBClassifier
from env import Env
from conector import Conector

# from tensorflow.keras import Model, Input
# from tensorflow.keras.layers import LSTM, Embedding, Dense
# from tensorflow.keras.layers import (
#     InputLayer,
#     TimeDistributed,
#     SpatialDropout1D,
#     Bidirectional,
# )
# from tensorflow import keras
# from sklearn.model_selection import train_test_split


# alias typing
# Model: TypeAlias = ActiveLearner
DataFrame: TypeAlias = pd.DataFrame
CrsMatrix: TypeAlias = csr_matrix
Vectorizer: TypeAlias = TfidfVectorizer


class ActiveLearning:
    def __init__(self) -> None:
        self.env = Env()
        self.model = None
        self.vectorizer: Vectorizer = None
        self.conector = Conector(self.env)
        self.word2idx = None
        self.tag2idx = None
        self.sentences = None
        self.max_len = 50
        self.num_words = 35179

        self.create_bag_of_words()

    def create_bag_of_words(self) -> None:
        with open("data/tag2idx.json", "r") as fp:
            self.tag2idx = json.load(fp)

        with open("data/word2idx.json", "r") as fp:
            self.word2idx = json.load(fp)

        self.num_words = max(self.word2idx.values())

    def sentence_integrate(self, data):
        agg_func = lambda s: [
            (w, p, t)
            for w, p, t in zip(
                s["Word"].values.tolist(),
                s["POS"].values.tolist(),
                s["Tag"].values.tolist(),
            )
        ]
        return data.groupby("Sentence").apply(agg_func).tolist()

    def data_preprocessing(self, df: DataFrame) -> tuple[Any, Any]:
        """
        Pré-processamento dos dados, retornando dados vetorizados
        e seus rótulos
        """
        self.create_bag_of_words()
        self.sentences = self.sentence_integrate(df)

        X = [[self.word2idx[w[0]] for w in s] for s in self.sentences]

        X = pad_sequences(
            maxlen=self.max_len,
            sequences=X,
            padding="post",
            value=self.num_words - 1,
        )

        y = [[self.tag2idx[w[2]] for w in s] for s in self.sentences]
        y = pad_sequences(
            maxlen=self.max_len,
            sequences=y,
            padding="post",
            value=self.tag2idx["O"],
        )

        return X, y

    def read_csv(self, name: str) -> DataFrame:
        """
        Lê um csv e o retorna em DataFrame
        """
        path = os.path.join("dataset", name)
        return pd.read_csv(path)

    def save_parquet(self, df: DataFrame, name: str) -> None:
        """
        Salva um DataFrame em parquet
        """

        path = os.path.join("data/datasets", name)
        path += ".parquet.gzip"
        df.to_parquet(path, engine="pyarrow", compression="gzip")

    def train(self, x_train, y_train) -> None:
        """
        Treina o modelo
        """

        self.model.teach(x_train, y_train)

    def get_most_uncertainty(self, data_unlabeled: DataFrame) -> DataFrame:
        """
        Pega os dados preditos mais incertos
        """
        data_unlabeled = data_unlabeled.fillna(method="ffill")
        unlabeled_sentences = self.sentence_integrate(data_unlabeled)
        print("Sentences criadas")

        X = [[self.word2idx[w[0]] for w in s] for s in unlabeled_sentences]
        X = pad_sequences(
            maxlen=self.max_len,
            sequences=X,
            padding="post",
            value=self.num_words - 1,
        )

        print("X criado")

        y = [[self.tag2idx[w[2]] for w in s] for s in unlabeled_sentences]
        y = pad_sequences(
            maxlen=self.max_len,
            sequences=y,
            padding="post",
            value=self.tag2idx["O"],
        )
        print("Y criado")

        predictions = self.model.predict(X)
        print("Predições criadas")
        print(predictions)
        # get the most uncertainty

        # return the most uncertainty
        # data = data_unlabeled.iloc[
        #     uncertainty.argsort()[: self._quantity_to_label]
        # ]

        # print(data.head())
        return data_unlabeled.iloc[: self._quantity_to_label]

        # pool = self.vectorizer.transform(data_unlabeled["text"])
        # query_idx = uncertainty_batch_sampling(
        #     self.model, pool, self._quantity_to_label
        # # )
        # data_unlabeled["prediction"] = self.model.predict(pool)
        # data_unlabeled["prediction"] = self.le.inverse_transform(
        #     data_unlabeled["prediction"]
        # )
        # print(data_unlabeled.head())

        # # create a new column with the uncertainty of the prediction
        # data_unlabeled["uncertaint"] = self.model.predict_proba(pool).max(
        #     axis=1
        # )
        # data_unlabeled["uncertaint"] = data_unlabeled["uncertaint"].apply(
        #     lambda x: 1 - x
        # )

        # print(data_unlabeled.head())
        # data_unlabeled = data_unlabeled.sort_values(
        #     by="uncertaint", ascending=False
        # )

        # return data_unlabeled.head(self.env.QUANTITY_TO_LABEL)

    def evaluate(self) -> None:
        """
        Salva as métricas (classification report e score) do modelo
        """

        df_test = self.read_csv("ner_dataset_test.csv")
        df_test = df_test.fillna(method="ffill")

        print("Evaluate on test data")

        x_test, y_test = self.data_preprocessing(df_test)

        results = self.model.evaluate(x_test, y_test, batch_size=128)
        print("test loss: {} ".format(results[0]))
        print("test accuracy: {} ".format(results[1]))

        history = {}
        with open("data/history.json", "r") as fp:
            history = json.load(fp)

        actual_data = {
            "loss": results[0],
            "accuracy": results[1],
        }

        history["data"].append(actual_data)

        with open("data/history.json", "w") as fp:
            json.dump(history, fp)

    def observer(self) -> None:
        """
        Verifica se há algum novo dado para ser importado do Label Studio
        """

        content = self.conector.get_tasks()
        # content = self.get_tasks()
        print(
            content.get("total_annotations"),
            type(content.get("total_annotations")),
        )
        if content.get("total_annotations") >= self.env.QUANTITY_TO_LABEL:
            print("New data to be imported")
            self.pipeline()
        else:
            print("No new data to be imported")

    def cron(self):
        """
        Cron job para verificar se tem algum dado novo para ser importado
        Acontece todos os dias a meia-noite
        """

        if self.env.DEBUG:
            schedule.every(3).seconds.do(self.observer)
        else:
            schedule.every().day.at("00:00").do(self.observer)
        while True:
            schedule.run_pending()
            time.sleep(1)

    def pipeline(self) -> None:
        """
        Pipeline completo de execução do fluxo e treinamento do modelo
        """
        print("Starting pipeline...")

        print("Importing data...")
        # df_train = self.data_import()
        df_train = self.conector.data_import()
        print("Data imported")

        print("Preprocessing data...")
        x_train, y_train = self.data_preprocessing(df_train)
        print("Data preprocessed")

        print("Training model...")
        self.train(x_train, y_train)
        print("Model trained")

        print("Evaluating model...")
        self.evaluate()
        print("Model evaluated")

        print("Saving model...")
        self.save("model", self.model)
        print("Model saved")

        print("Concatenating dataset...")
        self.concat_dataset(df_train, "train")
        print("Dataset concatenated")

        print("Getting most uncertainty data...")
        df_unlabeled = self.read_parquet("df_unlabeled")
        df_most_uncertainty = self.get_most_uncertainty(df_unlabeled)
        print("Most uncertainty data got")

        print("Exporting data to Label Studio...")
        # self.data_export(df_most_uncertainty)
        self.conector.data_export(df_most_uncertainty)
        print("Data exported to Label Studio")

        print("Pipeline finished")

    def concat_dataset(self, df_new: DataFrame, type_dataset: str) -> None:
        """
        Concatena um novo dataset com um já existente
        """

        name_dataset = f"df_{type_dataset}"
        df_old = self.read_parquet(name_dataset)
        df = pd.concat([df_old, df_new])
        self.save_parquet(df, name_dataset)

    def setup(self) -> None:
        """
        Configura versão inicial do fluxo do projeto
        """
        print("Loading model")

        self.model = load_model("data/model.h5")

        print("Evaluate")
        self.evaluate()

        print("Get most uncertainty")

        df_unlabeled = self.read_csv("ner_dataset_unlabeled.csv")
        df_most_uncertainty = self.get_most_uncertainty(df_unlabeled)

        print("Export")

        self.conector.data_export(df_most_uncertainty)

        # print("Save")

        # # Save model
        # self.save("model.h5", self.model)

        # self.cron()

    def save(self, name: str, obj: Any) -> None:
        """
        Salva qualquer tipo de dado
        """
        path = os.path.join("data", name)
        obj.save(path)
        # path += ".pkl"
        # pickle.dump(obj, open(path, "wb"))

    def load_state(self):
        """
        Carrega o estado do modelo
        """

        self.model = self.load("model")
        self.vectorizer = self.load("vectorizer")
        self.le = self.load("le")

    def load(self, name: str) -> Any:
        """
        Carrega qualquer tipo de dado
        """

        path = os.path.join("data", name)
        path += ".pkl"
        return pickle.load(open(path, "rb"))

    def run(self) -> None:
        print(f"Running with mode: { self.env.MODE}")
        if self.env.MODE == "INITIAL":
            self.setup()
        elif self.env.MODE == "TRIGGER":
            self.load_state()
            self.pipeline()
        else:
            self.load_state()
            self.cron()
