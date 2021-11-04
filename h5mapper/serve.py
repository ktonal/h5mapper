import dataclasses as dtc

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Union
import re
from torch._six import string_classes
import collections


__all__ = [
    'Setter',
    'Getter',
    'AsSlice',
    'AsFramedSlice',
    'GetId',
    'Input',
    'Target',
    'process_batch',
    'ProgrammableDataset',
]


@dtc.dataclass
class Setter:

    dim: int = 0
    after_item: bool = True

    def __post_init__(self):
        self.pre_slices = (slice(None),) * self.dim

    def __call__(self, data, item, value):
        slc = slice(item, item + value.shape[self.dim]) if self.after_item \
            else slice(item-value.shape[self.dim], item)
        data.data[self.pre_slices + (slc,)] = value
        return value.shape[self.dim]


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
        return proxy[proxy.refs[item]]


@dtc.dataclass
class AsSlice(Getter):
    """
    maps an ``item`` to a slice of data

    Parameters
    ----------
    dim : int
        the dimension to slice
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
    dim: int = 0
    shift: int = 0
    length: int = 1
    downsampling: int = 1

    def __post_init__(self):
        self.pre_slices = (slice(None),) * self.dim

    def __call__(self, proxy, item):
        i = item * self.downsampling
        slc = slice(i + self.shift, i + self.shift + self.length)
        return proxy[self.pre_slices + (slc, )]

    def __len__(self):
        return (self.n - (abs(self.shift) + self.length) + 1) // self.downsampling

    def shift_and_length_to_samples(self, frame_length, hop_length, center=False):
        extra = -hop_length if center else \
            ((frame_length // hop_length) - 1) * hop_length
        shift = self.shift * hop_length
        length = self.length * hop_length + extra
        return shift, length


@dtc.dataclass
class AsFramedSlice(AsSlice):
    dim: int = 0
    shift: int = 0  # in frames!
    length: int = 1  # in frames!
    frame_size: int = 1
    hop_length: int = 1
    center: bool = False
    pad_mode: str = 'reflect'
    downsampling: int = 1

    def __post_init__(self):
        super(AsFramedSlice, self).__post_init__()
        # convert frames to samples
        if self.hop_length != self.frame_size:
            _, self.length = self.shift_and_length_to_samples(
                self.frame_size, self.hop_length, self.center)

    def __call__(self, proxy, item):
        sliced = super(AsFramedSlice, self).__call__(proxy, item)
        if self.center:
            sliced = np.pad(sliced, int(self.frame_size // 2), self.pad_mode)
        return librosa.util.frame(sliced, self.frame_size, self.hop_length, axis=0)


@dtc.dataclass
class Input:
    """read and transform data from a specific key/proxy in a .h5 file"""
    data: Union[str, np.ndarray, "Proxy"] = ''
    getter: Getter = Getter()
    setter: Optional[Setter] = None
    transform: Callable[[np.ndarray], np.ndarray] = lambda x: x
    inverse_transform: Callable[[np.ndarray], np.ndarray] = lambda x: x
    to_tensor: bool = False
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        pass

    def get_object(self, file):
        return self.data if file is None or not isinstance(self.data, str) \
            else getattr(file, self.data)

    def __len__(self):
        return len(self.getter)

    def __call__(self, item, file=None):
        data = self.getter(self.get_object(file), item)
        if self.to_tensor:
            data = torch.from_numpy(data).to(self.device)
        return self.transform(data)

    def set(self, key, value):
        return self.setter(self.data, key, value)


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
    and should contain batch items (``h5m.Input`` or ``h5m.Target``).
    """

    def __init__(self, file, batch=tuple()):
        super(Dataset, self).__init__()
        self.file = file

        def cache_lengths(feat):
            # pass the lengths of the db features to the getters
            if feat.getter.n is None:
                if isinstance(feat.getter, GetId):
                    n = sum(feat.get_object(file).refs[()].astype(np.bool))
                else:
                    n = len(feat.get_object(file))
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
            return feat(item, self.file)

        return process_batch(self.batch, _is_batchitem, get_data)

    def __len__(self):
        return self.N

    def __del__(self):
        if hasattr(self.file, 'close'):
            self.file.close()