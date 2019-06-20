import sympl as sp
import numpy as np
from marble.state import AliasDict


class NotAColumnException(Exception):
    pass


class ColumnStore(sp.Monitor):
    """
    Stores single-column values as numpy arrays to later retrieve a timeseries.
    """

    def __init__(self, *args, **kwargs):
        super(ColumnStore, self).__init__(*args, **kwargs)
        self._column_lists = AliasDict()

    def store(self, state):
        """
        Store a given column state.

        Units and dimensions are assumed to be the same each time the state is
        stored. All arrays must be 0 or 1-dimensional.

        Args:
            state (dict): a state dictionary.
        """
        for name, array in state.items():
            if name == 'time':
                pass
            elif len(array.shape) > 1:
                raise NotAColumnException(
                    'array for {} is not a column, has shape {}, dims {}'.format(
                        name, array.shape, array.dims)
                )
            elif len(array.shape) == 1:
                self._column_lists[name] = self._column_lists.get(name, [])
                self._column_lists[name].append(array.values[None, :])
            elif len(array.shape) == 0:
                self._column_lists[name] = self._column_lists.get(name, [])
                self._column_lists[name].append(array.values[None])

    def __getitem__(self, item):
        return np.concatenate(self._column_lists[item], axis=0)
