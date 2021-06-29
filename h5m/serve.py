import dataclasses as dtc
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Callable
import re
from torch._six import string_classes
import collections


__all__ = [
    'AsSlice',
    'AsFramedSlice',
    'Input',
    'Target',
    'process_batch',
    'DefaultDataset',
]


@dtc.dataclass
class Getter:
    """
    base class for implementing data getter

    Parameters
    ----------

    Attributes
    ----------
    n : int or None
        the length of the underlying data
    """
    n: Optional[int] = dtc.field(default=None, init=False)

    def __call__(self, feat_data, item):
        """
        apply this instance's logic to get data from ``feat_data`` for a given ``item``

        Parameters
        ----------
        feat_data: [np.ndarray, torch.Tensor, mimikit.FeatureProxy]

        item: int
            the index emitted from a sampler

        Returns
        -------
        data: Any
            the examples corresponding to this item
        """
        return feat_data[item]

    def __len__(self):
        return self.n


@dtc.dataclass()
class AsSlice(Getter):
    """
    maps an ``item`` to a slice of data

    Parameters
    ----------
    shift : int
        the slice will start at the index `item + shift`
    length : int
        the length of the slice
    stride : int
        sub-sampling factor. Every `stride` datapoints `item` increases of `1`

    Examples
    --------

    .. testcode::

       import mimikit as mmk

       slicer = mmk.AsSlice(shift=2, length=3)
       data, item = list(range(10)), 2

       # now use it like a function :
       sliced = slicer(data, item)

       print(sliced)

    will output:

    .. testoutput::

       [4, 5, 6]
    """
    shift: int = 0
    length: int = 1
    stride: int = 1

    def __call__(self, feat_data, item):
        i = item * self.stride
        return feat_data[slice(i + self.shift, i + self.shift + self.length)]

    def __len__(self):
        return (self.n - (self.shift + self.length) + 1) // self.stride


@dtc.dataclass
class AsFramedSlice(AsSlice):
    frame_size: int = 1
    as_strided: bool = False

    def __call__(self, feat_data, item):
        sliced = super(AsFramedSlice, self).__call__(feat_data, item)
        if self.as_strided:
            if isinstance(sliced, np.ndarray):
                as_strided = lambda tensor: torch.as_strided(torch.from_numpy(tensor),
                                                             size=(self.length-self.frame_size+1, self.frame_size),
                                                             stride=(1, 1))
            else:
                as_strided = lambda tensor: torch.as_strided(tensor,
                                                             size=(self.length-self.frame_size+1, self.frame_size),
                                                             stride=(1, 1))

            with torch.no_grad():
                return as_strided(sliced)
        else:
            return sliced.reshape(-1, self.frame_size)


@dtc.dataclass
class Input:
    db_key: str = ''
    getter: Getter = Getter()
    transform: Callable = lambda x: x

    def __len__(self):
        return len(self.getter)


class Target(Input):
    """exactly equivalent to Input, just makes code simpler to read."""
    pass


np_str_obj_array_pattern = re.compile(r'[SaUO]')


def process_batch(batch, test=lambda x: False, func=lambda x: x):
    """
    recursively apply func to the elements of data if test(element) is True.
    This is used in DefaultDataset to process elements (Input or Target) packed in tuples, list, dict etc...
    """
    elem_type = type(batch)
    if test(batch):
        return func(batch)
    elif isinstance(batch, collections.abc.Mapping):
        return {key: process_batch(batch[key], test, func) for key in batch}
    elif isinstance(batch, tuple) and hasattr(batch, '_fields'):  # namedtuple
        return elem_type(*(process_batch(d, test, func) for d in batch))
    elif isinstance(batch, collections.abc.Sequence) and not isinstance(batch, string_classes):
        return [process_batch(d, test, func) for d in batch]
    else:
        return batch


def _is_batchitem(obj):
    return isinstance(obj, (Input, Target))


class DefaultDataset(Dataset):

    def __init__(self, file, batch=tuple()):
        super(Dataset, self).__init__()
        self.file = file
        # pass the lengths of the db features to the getters
        def cache_lengths(feat):
            if feat.getter.n is None:
                setattr(feat.getter, 'n', len(getattr(self.file, feat.db_key)))
            return feat

        self.batch = process_batch(batch, _is_batchitem, cache_lengths)

        # get the minimum length of all batchitems
        self.N = float('inf')

        def set_n_to_min(feat):
            self.N = min(len(feat), self.N)
            return feat

        process_batch(self.batch, _is_batchitem, set_n_to_min)

    def __getitem__(self, item):
        def get_data(feat):
            return feat.transform(feat.getter(getattr(self.file, feat.db_key), item))

        return process_batch(self.batch, _is_batchitem, get_data)

    def __len__(self):
        return self.N