import pickle
from typing import Any


class IOUtils:
    @staticmethod
    def pickle_dump_object(obj: Any, filepath: str) -> None:
        with open(filepath, 'wb') as file:
            pickle.dump(obj, file)

    @staticmethod
    def pickle_load_file(filepath: str) -> Any:
        with open(filepath, 'rb') as file:
            return pickle.load(file)
