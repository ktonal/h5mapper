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
    # transforms to use when unloading values
    __t__ = ()

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

    __t__ = (
        # getting a dict returns it filled with tensors
        partial(depth_first_apply, func=torch.from_numpy)
    )

    @property
    def attrs(self):
        return {}

    def load(self, source):
        return depth_first_apply(torch.load(source), lambda t: t.cpu().numpy())


class Image(Feature):
    __re__ = r"png$|jpeg$|"


class Sound(Feature):
    __re__ = r"wav$|aif$|aiff$|mp3$|mp4$|m4a$|"


class VShape(Feature):
    __t__ = (
        lambda d: d["arr"].reshape(*d["shape"])
    )

    def __init__(self, base_feat):
        self.base_feat = base_feat
        # preserve the base's config
        setattr(self, "__re__", base_feat.__re__)
        setattr(self, "__ds_kwargs__", base_feat.__ds_kwargs__)
        # chain the base's transform after our
        setattr(self, "__t__", (*self.__t__, *base_feat.__t__))

    @property
    def attrs(self):
        return self.base_feat.attrs

    def load(self, source):
        arr = self.base_feat.load(source)
        return {"arr": arr.reshape(-1), "shape": np.array(arr.shape)}


class Vocabulary(Feature):

    def __init__(self, derived_from):
        self.derived_from = derived_from
        self.D = Manager().dict()

    def load(self, source):
        items = set(source)
        self.D.update({x: i for x, i in zip(items, range(len(self.D), len(items)))})

    def after_create(self, db, feature_key):
        feat = db.get_feat(feature_key)
        x = np.array(list(self.D.keys()))
        i = np.array(list(self.D.values()))
        feat.add("xi", {"x": x, "i": i})
