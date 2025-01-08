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
    def drop_cells(column: str, sep: str = ">>", ratio: float = 0.1):
        """TODO
        
        Note: shuffles and drops (because of sample method)
        """
        column_as_list = column.split(sep)
        num_cells = len(column_as_list)
        num_samples = int(num_cells * ratio)
        
        return "".join(random.sample(column_as_list, num_samples))


if __name__ == "__main__":
    pass
