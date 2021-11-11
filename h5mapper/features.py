import h5py
import numpy as np
import imageio
import dill as pickle
import torch
import librosa
import os
import dataclasses as dtc
from functools import partial
try:
    from functools import cached_property
except ImportError:  # python<3.8
    def cached_property(f): return f

from multiprocess import Manager
import warnings

from .utils import depth_first_apply
from .crud import _load

warnings.filterwarnings("ignore", message="PySoundFile failed.")


__all__ = [
    'Feature',
    'Array',
    'Group',
    "TensorDict",
    "Image",
    "Sound",
    "VShape",
    "Vocabulary",
    "DirLabels",
    "FilesLabels",
]


class Feature:
    # re to match sources
    __re__ = r".*"
    # kwargs for h5py.create_dataset
    __ds_kwargs__ = {}
    # transforms to use at the array level
    __t__ = ()
    # transforms at the group level
    __grp_t__ = ()

    _proxy = None

    @property
    def attrs(self):
        return {}

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            # access from the class object
            return self
        # when accessed from an instance,
        # user expects a Proxy that should already be
        # in obj's __dict__
        if self.name not in obj.__dict__:
            raise RuntimeError(f"Feature '{self.name}' has not been properly attached to its parent object {obj},"
                               " it cannot mirror any h5 object.")
        proxy = obj.__dict__[self.name]
        self._proxy = proxy
        return proxy

    def __set__(self, obj, value):
        obj.__dict__[self.name] = value
        return value

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        if item in type(self._proxy).__dict__ or \
                (self._proxy is not None and item in self._proxy.__dict__):
            return getattr(self._proxy, item)
        else:
            raise AttributeError(f"object of type {self.__class__.__qualname__} has no attribute '{item}'")

    def load(self, source):
        raise NotImplementedError

    def after_create(self, db, feature_key):
        pass

    def __repr__(self):
        name = getattr(self, 'name', "UNK")
        return f"<Array '{name}'>"

    def __getstate__(self):
        state = self.__dict__.copy()
        if "__t__" in state:
            state.pop("__t__")
        if "__grp_t__" in state:
            state.pop("__grp_t__")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.update({"__t__": type(self).__t__})
        self.__dict__.update({"__grp_t__": type(self).__grp_t__})


class Array(Feature):

    def __init__(self, pattern=None, **ds_kwargs):
        if pattern is not None:
            self.__re__ = pattern
        if ds_kwargs:
            self.__ds_kwargs__.update(ds_kwargs)


class Group(Feature):

    def __init__(self, **features):
        for k, v in features.items():
            v.__set_name__(self, k)
            setattr(self, k, v)
        self.features = features

    @property
    def attrs(self):
        """concatenate all the features attrs in one dict"""
        return dict(list(item for f in self.features.values()
                         for item in f.attrs.items()))

    def load(self, source):
        return _load(source, self.features, guard_func=Feature.load)

    def after_create(self, db, feature_key):
        for feat in self.features.values():
            if getattr(type(feat), "after_create", Feature.after_create) != Feature.after_create:
                feat.after_create(db, feature_key + "/" + feat.name)


class TensorDict(Feature):
    __re__ = r".*ckpt$"
    __ds_kwargs__ = dict()

    __grp_t__ = (
        # getting a dict returns it filled with tensors
        partial(depth_first_apply, func=torch.from_numpy),
    )

    def __init__(self, state_dict={}):
        self.__ds_kwargs__ = {k: dict(compression="lzf",
                                      chunks=tuple(state_dict[k].shape))
                              for k in state_dict.keys()}

    @property
    def attrs(self):
        return {}

    def load(self, source):
        return self.format(torch.load(source))

    @staticmethod
    def format(state_dict):
        return depth_first_apply(state_dict, lambda t: np.atleast_1d(t.detach().cpu().numpy()))

    def save_hp(self, hp):
        self.h5_.attrs.update({"hp": np.void(pickle.dumps(hp))})

    def load_hp(self):
        return pickle.loads(self.h5_.attrs['hp'].tobytes())

    def load_checkpoint(self, module_cls, source, **overrides):
        hp = self.load_hp()
        hp.update(overrides)
        if "cls" in hp:
            hp.pop("cls")
        net = module_cls(**hp)
        net.load_state_dict(self.get(source))
        return net


class Image(Array):
    __re__ = r"png$|jpeg$"
    __ds_kwargs__ = dict()

    @property
    def attrs(self):
        return {}

    def load(self, source):
        img = imageio.imread(source)
        return img


@dtc.dataclass
class Sound(Array):
    __re__ = r"wav$|aif$|aiff$|mp3$|mp4$|m4a$"

    sr: int = 22050
    mono: bool = True
    normalize: bool = True

    @property
    def attrs(self):
        return dtc.asdict(self)

    def load(self, source):
        y = librosa.load(source, self.sr, self.mono)[0]
        if self.normalize:
            y = librosa.util.normalize(y, )
        return y


class VShape(Feature):
    __grp_t__ = (
        lambda d: d["arr"].reshape(*d["shape_"]),
    )

    def __init__(self, base_feat):
        self.base_feat = base_feat
        # preserve the base's config
        setattr(self, "__re__", base_feat.__re__)
        setattr(self, "__ds_kwargs__", base_feat.__ds_kwargs__)
        # chain the base's transform after our
        setattr(self, "__grp_t__", (*self.__grp_t__, *base_feat.__grp_t__))

    @property
    def attrs(self):
        return self.base_feat.attrs

    def load(self, source):
        arr = self.base_feat.load(source)
        return {"arr": arr.reshape(-1), "shape_": np.array(arr.shape)}


class Vocabulary(Feature):

    def __init__(self, derived_from):
        self.derived_from = derived_from
        self.V = Manager().dict()

    def load(self, source):
        # here `source` is the data loaded by `derived_from`
        if isinstance(source, np.ndarray):
            source = source.flat[:]
        items = {*source}
        self.V.update({x: i for x, i in zip(items, range(len(self.V), len(items)))})

    def after_create(self, db, feature_key):
        feat = db.get_proxy(feature_key)
        x = np.array(list(self.V.keys()))
        i = np.array(list(self.V.values()))
        # source "data" are the keys and values
        feat.add("data", {"x": x, "i": i})
        self.V = dict(self.V)

    @cached_property
    def dict(self):
        return dict(zip(self.i[:], self.x[:]))


class DirLabels(Feature):

    __ds_kwargs__ = dict(labels={}, dirs=dict(dtype=h5py.string_dtype(encoding='utf-8')))

    def __init__(self):
        self._d2i = Manager().dict()

    def load(self, source):
        direc = os.path.dirname(source)
        self._d2i.setdefault(direc, len(self._d2i))
        return {"labels": np.array([self._d2i[direc]]), "dirs": np.array([direc])}

    @cached_property
    def d2i(self):
        return {k: v for k, v in zip(self.dirs[:], self.labels[:])}

    @cached_property
    def i2d(self):
        return {v: k for k, v in zip(self.dirs[:], self.labels[:])}


class FilesLabels(Feature):
    """broadcast labels as source.shape[0] (the data of the feature it is derived from)"""

    def __init__(self, derived_from=""):
        self.derived_from = derived_from
        self.count = 0

    def load(self, source):
        res = np.ones((source.shape[0], ), dtype=np.int) * self.count
        self.count += 1
        return res

    @cached_property
    def f2i(self):
        db = self._proxy.owner
        return {k: v for k, v in db.index.items() if self._proxy.refs[v]}

    @cached_property
    def i2f(self):
        db = self._proxy.owner
        return {v: k for k, v in db.index.items() if self._proxy.refs[v]}


class String(Feature):
    __ds_kwargs__ = dict(dtype=h5py.string_dtype(encoding='utf-8'))
