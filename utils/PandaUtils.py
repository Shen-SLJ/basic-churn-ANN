from typing import Any

import pandas as pd


class PandaUtils:
    @staticmethod
    def dataframe_from_dict(dict_: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame([dict_])