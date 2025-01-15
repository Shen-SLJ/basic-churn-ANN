import pickle
from typing import Any


class IOUtils:
    @staticmethod
    def pickle_dump_object(obj: Any, filename: str) -> None:
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)

    @staticmethod
    def pickle_load_file(filename: str) -> Any:
        with open(filename, 'rb') as file:
            return pickle.load(file)
