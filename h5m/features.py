import h5py
import numpy as np
import torch
from functools import partial
from multiprocessing import Manager

from .utils import depth_first_apply
from .crud import _load


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
            raise RuntimeError(f"Feature '{self.name}' has not been properly attached to its parent object {obj},"
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
        return f"<Feature '{name}'>"

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


class StateDict(Feature):
    __re__ = r".*ckpt$"
    __ds_kwargs__ = dict()

    __grp_t__ = (
        # getting a dict returns it filled with tensors
        partial(depth_first_apply, func=torch.from_numpy),
    )

    def __init__(self, state_dict):
        self.keys = list(state_dict.keys())
        self.__ds_kwargs__ = {k: dict(compression="lzf", chunks=tuple(state_dict[k].shape))
                              for k in self.keys}

    @property
    def attrs(self):
        return {}

    def load(self, source):
        return depth_first_apply(torch.load(source), lambda t: t.cpu().numpy())


class Image(Feature):
    __re__ = r"png$|jpeg$|"
    __ds_kwargs__ = dict(
        dtype=np.dtype("uint8"),
    )


class Sound(Feature):
    __re__ = r"wav$|aif$|aiff$|mp3$|mp4$|m4a$|"


class VShape(Feature):
    __grp_t__ = (
        lambda d: d["arr"].reshape(*d["_shape"]),
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
        return {"arr": arr.reshape(-1), "_shape": np.array(arr.shape)}


class Vocabulary(Feature):

    def __init__(self, derived_from):
        self.derived_from = derived_from
        self.V = Manager().dict()

    def load(self, source):
        if isinstance(source, np.ndarray):
            source = source.flat[:]
        items = {*source}
        self.V.update({x: i for x, i in zip(items, range(len(self.V), len(items)))})

    def after_create(self, db, feature_key):
        feat = db.get_feat(feature_key)
        x = np.array(list(self.V.keys()))
        i = np.array(list(self.V.values()))
        # source "xi" is the dictionary
        feat.add("xi", {"x": x, "i": i})
        self.V = dict(self.V)