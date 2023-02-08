import json
import os
import pickle
import time
from typing import Any, TypeAlias

import nltk
import pandas as pd
import requests
import schedule
from modAL.batch import uncertainty_batch_sampling
from modAL.models.learners import ActiveLearner
from scipy.sparse._csr import csr_matrix
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from env import Env
from conector import Conector

# alias typing
Model: TypeAlias = ActiveLearner
DataFrame: TypeAlias = pd.DataFrame
CrsMatrix: TypeAlias = csr_matrix
Vectorizer: TypeAlias = TfidfVectorizer


class ActiveLearning:
    def __init__(self) -> None:
        self.env = Env()
        self.model: Model = None
        self.vectorizer: Vectorizer = None
        self.conector = Conector(self.env)

    def data_preprocessing(self, df: DataFrame) -> tuple[Any, Any]:
        """
        Pré-processamento dos dados, retornando dados vetorizados e seus rótulos
        """
        print(df.head())
        X = self.vectorizer.transform(df["text"])
        Y = self.le.transform(df["label"])
        # Y = self.le.transform(df["class"])
        return X, Y

    def read_parquet(self, name: str) -> DataFrame:
        """
        Lê um parquet e o retorna em DataFrame
        """

        path = os.path.join("dataset", name)
        # path = os.path.join("data/datasets", name)
        path += ".parquet.gzip"
        return pd.read_parquet(path, engine="pyarrow")

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

    def get_most_uncertainty_pandas(
        self, data_unlabeled: DataFrame
    ) -> DataFrame:
        """
        Pega os dados preditos mais incertos
        """
        pool = self.vectorizer.transform(data_unlabeled["text"])
        # query_idx = uncertainty_batch_sampling(
        #     self.model, pool, self._quantity_to_label
        # )
        data_unlabeled["prediction"] = self.model.predict(pool)
        data_unlabeled["prediction"] = self.le.inverse_transform(
            data_unlabeled["prediction"]
        )
        print(data_unlabeled.head())

        # create a new column with the uncertainty of the prediction
        data_unlabeled["uncertaint"] = self.model.predict_proba(pool).max(
            axis=1
        )
        data_unlabeled["uncertaint"] = data_unlabeled["uncertaint"].apply(
            lambda x: 1 - x
        )

        print(data_unlabeled.head())
        data_unlabeled = data_unlabeled.sort_values(
            by="uncertaint", ascending=False
        )

        return data_unlabeled.head(self.env.QUANTITY_TO_LABEL)

    def evaluate(self) -> None:
        """
        Salva as métricas (classification report e score) do modelo
        """

        df_test = self.read_parquet("df_test")

        x_test, y_test = self.data_preprocessing(df_test)
        y_pred = self.model.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        score = self.model.score(x_test, y_test)
        report.update({"score": score})

        history_metrics = json.load(open("data/history_metrics.json", "r"))

        report.update({"iteration": len(history_metrics)})

        history_metrics.append(report)

        json.dump(history_metrics, open("data/history_metrics.json", "w"))

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
        Seta o modelo e o vetorizador
        """

        # Model
        print("Setup")

        # Stop Words
        print("Stopwords")
        stopwords = nltk.corpus.stopwords.words("portuguese")

        # Vectorizer
        print("Vectorizer")
        self.vectorizer = TfidfVectorizer(
            stop_words=stopwords, max_df=0.90, min_df=0.15
        )
        df_unlabeled = self.read_parquet("df_unlabeled")

        print("Label Encoder")
        # classes = json.load(open("data/classes.json", "r"))
        self.le = preprocessing.LabelEncoder()
        # self.le.fit(classes)
        self.le.fit(["Negativo", "Positivo"])
        # self.le.fit(["cacm", "cisi", "cran", "med"])

        self.vectorizer.fit(df_unlabeled["text"])

        print("Train")
        df_train = self.read_parquet("df_train")
        x_train, y_train = self.data_preprocessing(df_train)

        self.model = ActiveLearner(
            estimator=XGBClassifier(), X_training=x_train, y_training=y_train
        )

        print("Evaluate")
        self.evaluate()
        print("Get most uncertainty")
        # df_most_uncertainty = self.get_most_uncertainty(df_unlabeled)
        df_most_uncertainty = self.get_most_uncertainty_pandas(df_unlabeled)

        print("Export")

        self.conector.data_export(df_most_uncertainty)
        # self.data_export(df_most_uncertainty)
        print("Save")
        # Save vectorizer
        self.save("vectorizer", self.vectorizer)

        # Save model
        self.save("model", self.model)

        self.save("le", self.le)

        self.cron()

    def save(self, name: str, obj: Any) -> None:
        """
        Salva qualquer tipo de dado
        """

        path = os.path.join("data", name)
        path += ".pkl"
        pickle.dump(obj, open(path, "wb"))

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
