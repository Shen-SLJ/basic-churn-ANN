import pickle
from typing import Any

from utils.PathUtils import PathUtils


class IOUtils:
    @staticmethod
    def pickle_dump_object(obj: Any, filepath: str) -> None:
        with open(PathUtils.abs_path_from_project_root_path(filepath), 'wb') as file:
            pickle.dump(obj, file)

    @staticmethod
    def pickle_load_file(filepath: str) -> Any:
        with open(PathUtils.abs_path_from_project_root_path(filepath), 'rb') as file:
            return pickle.load(file)
