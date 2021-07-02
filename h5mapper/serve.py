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
    'GetId',
    'Input',
    'Target',
    'process_batch',
    'ProgrammableDataset',
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

    def __call__(self, proxy, item):
        """
        apply this instance's logic to get data from ``proxy`` for a given ``item``

        Parameters
        ----------
        proxy: h5m.Proxy
            the proxy to read from
        item: int
            the index emitted from a sampler

        Returns
        -------
        data: Any
            the data corresponding to this item
        """
        return proxy[item]

    def __len__(self):
        return self.n


class GetId(Getter):

    def __call__(self, proxy, item):
        # TODO : this should index directly into the array of
        #  valid ids for this proxy (owners may have more ids than
        #  any of their proxies...)
        return proxy.get(proxy.owner.refs.index[item])


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

       import h5mapper as h5m

       slicer = h5m.AsSlice(shift=2, length=3)
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

    def __call__(self, proxy, item):
        i = item * self.stride
        return proxy[slice(i + self.shift, i + self.shift + self.length)]

    def __len__(self):
        return (self.n - (self.shift + self.length) + 1) // self.stride


@dtc.dataclass
class AsFramedSlice(AsSlice):
    frame_size: int = 1
    as_strided: bool = False

    def __call__(self, proxy, item):
        sliced = super(AsFramedSlice, self).__call__(proxy, item)
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
    """read and transform data from a specific key/proxy in a .h5 file"""
    key: str = ''
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
    This is used in ProgrammableDataset to process elements (Input or Target) packed in tuples, list, dict etc...
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


class ProgrammableDataset(Dataset):
    """
    Dataset whose __getitem__ method is specified by a batch object passed to its constructor.

    The batch object can be of any type supported by torch's default collate function (Mapping, Sequence, etc.)
    and should contain "batch-items" (``h5m.Input`` or ``h5m.Target``).
    """

    def __init__(self, file, batch=tuple()):
        super(Dataset, self).__init__()
        self.file = file

        def cache_lengths(feat):
            # pass the lengths of the db features to the getters
            if feat.getter.n is None:
                if isinstance(feat.getter, GetId):
                    n = sum(getattr(file, feat.key).refs[()].astype(np.bool))
                else:
                    n = len(getattr(self.file, feat.key))
                setattr(feat.getter, 'n', n)
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
            return feat.transform(feat.getter(getattr(self.file, feat.key), item))

        return process_batch(self.batch, _is_batchitem, get_data)

    def __len__(self):
        return self.N

    def __del__(self):
        self.file.close()