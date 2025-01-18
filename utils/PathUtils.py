from definitions import ROOT_DIR


class PathUtils:
    @staticmethod
    def to_abs_path(project_root_path: str) -> str:
        return f"{ROOT_DIR}/{project_root_path}"
