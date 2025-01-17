import random


class Augmenter:
    """TODO"""

    @staticmethod
    def shuffle_row(column: str, sep: str = ">>") -> str:
        """TODO"""
        column_as_list = column.split(sep)
        random.shuffle(column_as_list)
        return "".join(column_as_list)

    @staticmethod
    def drop_cells(column: str, sep: str = " << ", ratio: float = 0.1):
        """TODO
        
        Note: drops random cells
        """
        assert 0.0 <= ratio <= 1.0
        column_as_list = column.split(sep)
        num_cells = len(column_as_list)
        num_samples = num_cells - int((num_cells * ratio))

        ids = [i for i in range(num_cells)]
        sampled_ids = random.sample(ids, num_samples)
        result = [column_as_list[i] for i in range(num_cells) if i in sampled_ids]

        return sep.join(result)
