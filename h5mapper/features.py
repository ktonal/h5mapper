import numpy as np
import torch
import librosa
import imageio
import os
import dataclasses as dtc
from functools import partial
from multiprocessing import Manager
import warnings

from .utils import depth_first_apply
from .crud import _load

warnings.filterwarnings("ignore", message="PySoundFile failed.")


class Feature:
    # re to match sources
    __re__ = r".*"
    # kwargs for h5py.create_dataset
    __ds_kwargs__ = {}
    # transforms to use at the array level
    __t__ = ()
    # transforms at the group level
    __grp_t__ = ()

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
            raise RuntimeError(f"Array '{self.name}' has not been properly attached to its parent object {obj},"
                               " it cannot mirror any h5 object.")
        proxy = obj.__dict__[self.name]
        return proxy

    def __set__(self, obj, value):
        obj.__dict__[self.name] = value
        return value

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
    pass


class Group(Array):

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
        return _load(source, self.features, guard_func=Array.load)

    def after_create(self, db, feature_key):
        for feat in self.features.values():
            if getattr(type(feat), "after_create", Array.after_create) != Array.after_create:
                feat.after_create(db, feature_key + "/" + feat.name)


class TensorDict(Array):
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


class Image(Array):
    __re__ = r"png$|jpeg$"
    __ds_kwargs__ = dict(
    )

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


class VShape(Array):
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


class Vocabulary(Array):

    def __init__(self, derived_from):
        self.derived_from = derived_from
        self.V = Manager().dict()

    def load(self, source):
        if isinstance(source, np.ndarray):
            source = source.flat[:]
        items = {*source}
        self.V.update({x: i for x, i in zip(items, range(len(self.V), len(items)))})

    def after_create(self, db, feature_key):
        feat = db.get_proxy(feature_key)
        x = np.array(list(self.V.keys()))
        i = np.array(list(self.V.values()))
        # source "xi" is the dictionary
        feat.add("xi", {"x": x, "i": i})
        self.V = dict(self.V)


class DirLabels(Array):

    def __init__(self):
        self.d2i = Manager().dict()

    def load(self, source):
        direc = os.path.split(source.strip("/"))[0]
        self.d2i.setdefault(direc, len(self.d2i))
        return np.array([self.d2i[direc]])

    def after_create(self, db, feature_key):
        self.d2i = dict(self.d2i)
        self.i2d = {v: k for k, v in self.d2i.items()}