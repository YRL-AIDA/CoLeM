from typing import Optional

from torch.utils.data import Dataset

import pandas as pd
import glob


class TableDataset(Dataset):
    """Wrapper class over the dataset.

    Args:
        data_dir: Path to directory, where dataset `.csv` files was placed.
        sep: csv separator.
        engine: Parser engine to use.
        quotechar: Character used to denote the start and end of a quoted item.
        on_bad_lines: Specifies what to do upon encountering a bad line.
        num_rows: Amount of how many rows to read per `.csv` file, if None read all rows.
        transform: Optional transform to be applied on a sample
        target_transform: Optional transform to be applied on a target.
    """
    def __init__(
            self,
            data_dir: str = "data/",
            sep: str = "⇧",
            engine: str = "python",
            quotechar: str = '"',
            on_bad_lines: str = "warn",
            num_rows: Optional[int] = None,
            transform=None,
            target_transform=None,
    ):
        
        self.df = self._read_multiple_csv(
            data_dir=data_dir,
            sep=sep,
            engine=engine,
            quotechar=quotechar,
            on_bad_lines=on_bad_lines,
            num_rows=num_rows
        )
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]["column_data"]

    def _read_multiple_csv(
            self,
            data_dir: str,
            sep: str,
            num_rows: Optional[int],
            engine: str,
            quotechar: str,
            on_bad_lines: str
    ) -> pd.DataFrame:
        """Read dataframe from multiple csv files.

        If dataset was split into multiple files, it will be concatenated. Dataset is stored
        in a pd.Dataframe instance.

        Args:
            data_dir: Path to directory, where dataset .csv files was placed.
            sep: csv separator character.
            num_rows: Amount of how many rows to read per .csv file, if None read all rows.
            engine: Parser engine to use.
            quotechar: Character used to denote the start and end of a quoted item.
            on_bad_lines: Specifies what to do upon encountering a bad line.

        Returns:
            pd.Dataframe: Entire dataset as a dataframe.
        """

        df_list = []
        chunks = glob.glob(data_dir + "*.csv")
        assert len(chunks) > 0

        for filename in chunks:
            df = pd.read_csv(
                filepath_or_buffer=filename,
                sep=sep,
                engine=engine,
                quotechar=quotechar,
                on_bad_lines=on_bad_lines,
                nrows=num_rows
            )
            df_list.append(df)
        df = pd.concat(df_list, axis=0)
        df.reset_index(inplace=True, drop=True)
        return df
