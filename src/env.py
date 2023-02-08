import os

from dotenv import load_dotenv


class Env:
    LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST", "")
    LABEL_STUDIO_PORT = os.getenv("LABEL_STUDIO_PORT", "")
    LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN", "")
    QUANTITY_TO_LABEL = int(os.getenv("QUANTITY_TO_LABEL", ""))
    PROJECT_ID = int(os.getenv("PROJECT_ID", ""))
    MODE = os.getenv("MODE", "")
    DEBUG = os.getenv("DEBUG", "false")

    def __str__(self) -> str:
        return f"Envs({self.__dict__})"

    def __repr__(self) -> str:
        return f"Envs({self.__dict__})"

    def __dict__(self) -> dict:
        return {
            "LABEL_STUDIO_HOST": self.LABEL_STUDIO_HOST,
            "LABEL_STUDIO_PORT": self.LABEL_STUDIO_PORT,
            "LABEL_STUDIO_TOKEN": self.LABEL_STUDIO_TOKEN,
            "QUANTITY_TO_LABEL": self.QUANTITY_TO_LABEL,
            "PROJECT_ID": self.PROJECT_ID,
            "MODE": self.MODE,
            "DEBUG": self.DEBUG,
        }
