import random


class Augmenter:
    """Tabular data augmenter"""

    @staticmethod
    def shuffle_rows(column: str, sep: str) -> str:
        """Shuffle random rows in a given column.

        Args:
            column: table column as string.
            sep: separates cells in a column as string

        Returns:
            str: column as string with shuffled rows.
        """
        column_as_list = column.split(sep)
        random.shuffle(column_as_list)
        return "".join(column_as_list)

    @staticmethod
    def drop_cells(column: str, sep: str, ratio: float) -> str:
        """Drop random cells in a given column.

        Args:
            column: table column as string.
            sep: separates cells in a column as string
            ratio: cells removal ratio.
        
        Returns:
            str: column as string with deleted cells.
        """
        assert 0.0 <= ratio <= 1.0
        column_as_list = column.split(sep)
        num_cells = len(column_as_list)
        num_samples = num_cells - int((num_cells * ratio))

        ids = [i for i in range(num_cells)]
        sampled_ids = random.sample(ids, num_samples)
        result = [column_as_list[i] for i in range(num_cells) if i in sampled_ids]

        return sep.join(result)
