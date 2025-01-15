from definitions import ROOT_DIR


class PathUtils:
    @staticmethod
    def abs_path_from_project_root_path(path: str) -> str:
        return f"{ROOT_DIR}/{path}"
