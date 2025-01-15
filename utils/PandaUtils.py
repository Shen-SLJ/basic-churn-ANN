from typing import Any

import pandas as pd


class PandaUtils:
    @staticmethod
    def convert_dict_to_dataframe(dict_: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(dict_)