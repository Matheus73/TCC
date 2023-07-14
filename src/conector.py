from env import Env
import pandas as pd
from typing import Any, TypeAlias
import requests
import json

DataFrame: TypeAlias = pd.DataFrame


class Conector:
    def __init__(self, env: Env):
        self.env = env
        self._label_studio_base_url = (
            f"http://{self.env.LABEL_STUDIO_HOST}:{self.env.LABEL_STUDIO_PORT}"
        )
        self.headers = {
            "Authorization": f"Token {self.env.LABEL_STUDIO_TOKEN}"
        }

    def data_import(self) -> DataFrame:
        """
        Exporta dados rotulados de um projeto do Label Studio
        """

        url = f"{self._label_studio_base_url}/api/projects/{self.env.PROJECT_ID}/export"  # noqa E501

        params = {
            "export_type": "JSON",
            "download_all_tasks": "False",
            "download_resources": "True",
        }
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
        except Exception as e:
            print(str(e))
        else:
            content = json.loads(response.content)
            return self.parse_import(content)

    def parse_import(self, content: dict[Any, Any]) -> DataFrame:
        """
        Trata um JSON de tarefas exportadas do Label Studio,
        transformando-o em um DataFrame com as colunas essenciais para o fluxo
        """

        exported_data = []

        # O dataset é composto por: texto, anotação e classe real.
        data = []
        for row in content:
            data.append(self.get_sentence_data(row))

        # Criar um DataFrame com os dados
        sentences_numbers = []
        for i in range(len(content)):
            sentences_numbers.append(
                [f"Sentence: {len(content[i]) + 10000}"] * len(content[i])
            )

        df = pd.DataFrame(data, columns=["Sentence", "Word", "Tag"])

        df = pd.DataFrame(exported_data)
        print(df.head())

        return df

    def get_sentence_data(self, data: dict):
        text = data["data"]["text"]
        tokens = text.split()

        # Criar uma lista de labels vazias para os tokens
        labels = ["O"] * len(tokens)

        # Preencher as labels com base nas anotações do JSON
        for annotation in data["annotations"]:
            for result in annotation["result"]:
                start = result["value"]["start"]
                end = result["value"]["end"]
                label = result["value"]["labels"][0]

                for i, token in enumerate(tokens):
                    token_start = text.find(token)
                    token_end = token_start + len(token)

                    if start <= token_start and end >= token_end:
                        labels[i] = label

        return [tokens, labels]

    def data_export(self, df: DataFrame) -> None:
        """
        Importa dados (em CSV) para um projeto do Label Studio.
        Executa a função de limpar tarefas antes da importação
        """
        # saving a data frame to a buffer (same as with a regular file):
        print("Exportando dados para o Label Studio")
        self.clear_tasks()
        csv = df.to_csv(index=False, sep=",")
        try:
            url = f"{self._label_studio_base_url}/api/projects/{int(self.env.PROJECT_ID)}/import"  # noqa E501
            files = {"data.csv": csv}

            response = requests.post(url, headers=self.headers, files=files)
            response.raise_for_status()
        except Exception as e:
            print(f"Falha ao exportar os dados {str(e)}")
            raise

    def get_tasks(self) -> Any:
        """
        Pega todas as tarefas presentes em um projeto do Label Studio
        """

        try:
            url = f"{self._label_studio_base_url}/api/tasks"
            params = {"project": self.env.PROJECT_ID}
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
        except Exception as e:
            print(str(e))
            raise
        else:
            content = response.json()
            return content

    def clear_tasks(self):
        """
        Limpa as tarefas de um projeto no Label Studio
        """

        tasks = self.get_tasks()
        tasks_id = [task["id"] for task in tasks["tasks"]]
        print(tasks_id)
        self.delete_task(tasks_id)

    def delete_task(self, tasks_id: list[int]):
        """
        Deleta as tarefas do Label Studio
        """

        print("Cleaning tasks...")
        try:
            for id in tasks_id:
                url = f"{self._label_studio_base_url}/api/tasks/{id}"
                response = requests.delete(url, headers=self.headers)
                response.raise_for_status()
        except Exception as e:
            print(str(e))
            raise
