from typing import Optional

from torch.utils.data import Dataset

import pandas as pd
import glob


class TableDataset(Dataset):
    """Wrapper class over the dataset.

    Args:
        data_dir: path to directory, where dataset .csv files placed.
        num_rows: amount of how many rows to read per .csv file, if None read all rows.
        transform: Optional transform to be applied on a sample
        target_transform: Optional transform to be applied on a target.
    """
    def __init__(
            self,
            data_dir: str,
            num_rows: Optional[int],
            file_name=None,
            transform=None,
            target_transform=None,
    ):
        # Read dataset .csv files
        if file_name:
            self.df = pd.read_csv(
                data_dir + file_name,
                sep="⇧",
                engine="python",
                quotechar='"',
                on_bad_lines="warn",
                nrows=num_rows if num_rows is not None else None
            )
        else:
            self.df = self.read_multiple_csv(data_dir, num_rows)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "table_id": self.df.iloc[idx]["table_id"],
            "header": self.df.iloc[idx]["column_header"],
            "data": self.df.iloc[idx]["column_data"],
        }

    def read_multiple_csv(self, data_dir: str, num_rows: Optional[int] = None) -> pd.DataFrame:
        """Read dataframe from multiple csv files.

        If dataset was split into multiple files, it will be concatenated. Dataset is stored
        in pd.Dataframe instance.

        Args:
            data_dir: path to directory, where dataset .csv files placed.
            num_rows: amount of how many rows to read per .csv file, if None read all rows.

        Returns:
            pd.Dataframe: Entire dataset as dataframe.
        """

        df_list = []
        num_chunks = len(glob.glob(data_dir + "data_*.csv"))
        if num_chunks > 1:
            for i in range(num_chunks):
                df = pd.read_csv(
                    data_dir + f"data_{i}.csv",
                    sep="⇧", # TODO: move me to config!
                    engine="python",
                    quotechar='"',
                    on_bad_lines="warn",
                    nrows=num_rows if num_rows is not None else None
                )
                df_list.append(df)
            return pd.concat(df_list, axis=0)
        
        return pd.read_csv(
            data_dir + "data.csv",
            sep="⇧",
            engine="python",
            quotechar='"',
            on_bad_lines="warn",
            nrows=num_rows if num_rows is not None else None
        )


if __name__ == "__main__":
    pass
    # TODO
