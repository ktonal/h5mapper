import dataclasses as dtc

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Union
import re
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

    n: Optional[int] = dtc.field(default=None, init=False)

    def __call__(self, proxy, item):
        X = proxy[item]
        return X.copy() if isinstance(X, np.ndarray) else X

    def __len__(self):
        return self.n


class GetId(Getter):

    def __call__(self, proxy, item):
        X = proxy[proxy.refs[item]]
        return X.copy() if isinstance(X, np.ndarray) else X


@dtc.dataclass
class AsSlice(Getter):

    dim: int = 0
    shift: int = 0
    length: int = 1
    downsampling: int = 1

    def __post_init__(self):
        self.pre_slices = (slice(None),) * self.dim

    def __call__(self, proxy, item):
        i = item * self.downsampling
        slc = slice(i + self.shift, i + self.shift + self.length)
        # !important!: .copy() prevent memory leaks in torch's Dataloader
        # see https://github.com/h5py/h5py/issues/2010
        X = proxy[self.pre_slices + (slc, )]
        return X.copy() if isinstance(X, np.ndarray) else X

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
        if isinstance(sliced, np.ndarray):
            return librosa.util.frame(sliced, frame_length=self.frame_size, hop_length=self.hop_length, axis=0)
        else:
            return sliced.unfold(0, self.frame_size, self.hop_length)


@dtc.dataclass
class Input:
    """read and transform data from a specific key/proxy in a .h5 file"""
    data: Union[str, np.ndarray, "Proxy"] = None
    getter: Getter = Getter()
    setter: Optional[Setter] = None
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    to_tensor: bool = False

    def __post_init__(self):
        pass

    def get_object(self, file):
        return self.data if file is None or not isinstance(self.data, str) \
            else getattr(file, self.data)

    def __len__(self):
        return len(self.getter)

    def __call__(self, item, file=None):
        data = self.getter(self.data, item)
        if self.to_tensor:
            data = torch.from_numpy(data)
        return self.transform(data) if self.transform is not None else data

    def set(self, key, value):
        return self.setter(self.data, key, value)


@dtc.dataclass
class Target:
    data: Union[str, np.ndarray, "Proxy"] = ''
    setter: Setter = Setter()
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def __call__(self, item, value, file=None):
        value = self.transform(value) if self.transform is not None else value
        return self.setter(self.data, item, value)


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
    elif isinstance(batch, collections.abc.Sequence) and not isinstance(batch, (str, bytes)):
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
        self.N = float('inf')

        def initialize_items(item: Union[Input, Target]):
            if isinstance(item.data, str) and self.file is not None:
                item.data = getattr(self.file, item.data)
            if item.getter.n is None:
                if isinstance(item.getter, GetId):
                    # will raise if feat is not a proxy...
                    n = sum(item.data.refs[()].astype(bool))
                else:
                    n = len(item.data)
                setattr(item.getter, 'n', n)
            self.N = min(len(item), self.N)
            return item

        self.batch = process_batch(batch, _is_batchitem, initialize_items)

    def __getitem__(self, item):
        def get_data(feat):
            return feat(item)
        return process_batch(tuple(self.batch), _is_batchitem, get_data)

    def __len__(self):
        return self.N

    def __del__(self):
        if hasattr(self.file, 'close'):
            self.file.close()
