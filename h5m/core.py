import h5py
import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool
from concurrent.futures import ThreadPoolExecutor
import os
from functools import partial

from .crud import *


class Feature:
    __ext__ = {"any"}
    __ds_kwargs__ = {}

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
        raise NotImplementedError


class Proxy:

    def __getattr__(self, attr):
        if attr in self.__dict__:
            dic = self.__dict__
        # access the feature's params
        elif attr in self.attrs:
            dic = self.attrs
        # access the feature's methods etc.
        elif self.feature is not None and attr in self.feature.__dict__:
            dic = self.feature.__dict__
        else:
            raise AttributeError(f"object of type {self.__class__.__qualname__} has no attribute '{attr}'")
        return dic[attr]

    @classmethod
    def from_group(cls, owner, feature, group):
        # print("INSTANTIATING", group.name)
        attrs = {k: v if v is not H5_NONE else None for k, v in group.attrs.items()}
        # we "lift up" x/__arr__ or x/__df__ to attributes named "x"
        if NP_KEY in group.keys():
            root = Proxy(owner, feature, group.name, NP_KEY, attrs)
            if SRC_KEY in group.keys():
                refs = Proxy(owner, feature, group.name + "/" + SRC_KEY, REF_KEY)
                ids = Proxy(owner, feature, group.name + "/" + SRC_KEY, ID_KEY)
                setattr(root, "refs", refs)
                setattr(root, "ids", ids)
        elif DF_KEY in group.keys():
            root = Proxy(owner, feature, group.name, DF_KEY, attrs)
        else:
            root = Proxy(owner, feature, group.name, "", attrs)

        # recursively add children as attributes before returning the object
        for k in group.keys():
            if isinstance(group[k], h5py.Group) and k not in PRIVATE_GRP_KEYS:
                child = Proxy.from_group(owner, feature, group[k])
            else:
                child = Proxy(owner, feature, group.name, "/"+k)
            # print("ATTACHING", k, child, "TO", root)
            setattr(root, k, child)
        return root

    def __init__(self, owner, feature, group_name, key="", attrs={}):
        self.h5_file = owner.h5_file
        self.name = "/".join([group_name, key.strip("/")])
        self.owner = owner
        self.feature = feature
        self.attrs = attrs
        self.is_group = not bool(key)
        if key == DF_KEY:
            self.kind = 'pd'
        else:
            self.kind = 'h5py'

    def handler(self):
        return self.owner.handler(self.kind)

    def getgrp(self, item):
        if isinstance(item, str):
            # item is a source name
            mask = self.src.id[:] == item
            refs, ks = self.src.ref[mask], self.src.key[mask]
            out = {}
            for ref, k in zip(refs, ks):
                if ref:  # only arrays have
                    o = getattr(self, k)[ref]
                else:
                    if getattr(self, k).is_group:
                        o = getattr(self, k)[item]
                    else:  # dataframe
                        o = getattr(self, k)[:].loc[item]
                out[k] = o
            return out
        elif isinstance(item, tuple):
            # (source name(s), key(s))
            return None
        return None

    def __getitem__(self, item):
        if self.is_group:
            return self.getgrp(item)
        if self.owner.keep_open:
            return self.handler()[self.name][item]
        with self.handler() as f:
            rv = f[self.name][item]
        return rv

    def __setitem__(self, item, value):
        if self.owner.keep_open:
            self.handler()[self.name][item] = value
        else:
            with self.handler() as f:
                f[self.name][item] = value

    def __repr__(self):
        return f"<Proxy {self.name}>"


class Database:

    def __init__(self, h5_file, mode="r", keep_open=False):
        self.h5_file = h5_file
        self.mode = mode
        self.keep_open = keep_open
        self._f = None
        self._store = None
        with self.handler('h5py', 'r') as f:
            # f.visit(print)
            for k in f.keys():
                feature = getattr(type(self), k, setattr(self, k, Feature()))
                self.__dict__[k] = Proxy.from_group(self, feature, f[k])

    @classmethod
    def create(cls,
               h5_file,
               sources,
               mode="w",
               n_workers=cpu_count() * 2,
               parallelism='mp',
               keep_open=False
               ):
        # get schema from the class attributes
        schema = {attr: val for attr, val in cls.__dict__.items() if isinstance(val, Feature)}
        # create two separate files for arrays and dataframes
        f = h5py.File(h5_file, mode)
        store = pd.HDFStore("/tmp/h5m-store.h5", mode='w')

        # create groups from schema and write attrs
        groups = {key: f.create_group(key) if key not in f else f[key] for key in schema.keys()}
        for key, grp in groups.items():
            for k, v in schema[key].attrs.items():
                grp.attrs[k] = v if v is not None else H5_NONE
        f.flush()

        # initialize ds_kwargs from schema
        ds_kwargs = {key: getattr(feature, "__ds_kwargs__", {}) for key, feature in schema.items()}
        # get flavour of parallelism
        if parallelism == 'mp':
            executor = Pool(n_workers)
        elif parallelism == 'future':
            executor = ThreadPoolExecutor(n_workers)
        else:
            raise ValueError(f"parallelism must be one of ['mp', 'future']. Got '{parallelism}'")

        # run loading routine
        n_sources = len(sources)
        batch_size = n_workers * 4
        for i in range(1 + n_sources // batch_size):
            start_loc = max([i * batch_size, 0])
            end_loc = min([(i + 1) * batch_size, n_sources])
            this_sources = sources[start_loc:end_loc]
            results = executor.map(partial(_load, schema=schema), this_sources)
            # write results
            for n, res in enumerate(results):
                for key, data in res.items():
                    if data is None:
                        continue
                    _add.source(this_sources[n], data, groups[key], ds_kwargs[key], store)
                    f.flush()
                    store.flush()

        if parallelism == 'mp':
            executor.close()
            executor.join()
        store.close()

        # copy dataframes to master file
        tmp = h5py.File("/tmp/h5m-store.h5", 'r')
        _add.groups(f, tmp.values())
        tmp.close()
        f.flush()

        # run after_create

        # re instantiate store
        store = pd.HDFStore("/tmp/h5m-store.h5", mode='a')

        # loop through the features
        db = cls(h5_file, mode="r+", keep_open=False)
        has_changed = {}
        for key, feature in schema.items():
            if getattr(type(feature), "after_create", Feature.after_create) != Feature.after_create:
                data = feature.after_create(db, key)
                if isinstance(data, np.ndarray):
                    _add.array(groups[key], key + '/data', data, ds_kwargs[key])
                    # no source ref...
                    f.flush()
                elif isinstance(data, pd.DataFrame):
                    data.to_hdf(store, key, mode='a', append=False, format='table')
                    store.flush()
                    has_changed[key] = True

        store.close()
        # copy dataframes to master file one more time
        tmp = h5py.File("/tmp/h5m-store.h5", 'r')
        _add.groups(f, [tmp[k] for k in has_changed.keys()])
        tmp.close()
        f.flush()
        f.close()
        os.remove("/tmp/h5m-store.h5")
        # voila!
        return cls(h5_file, mode if mode != 'w' else "r+", keep_open)

    def handler(self, kind, mode=None):
        """
        """
        mode = mode if mode is not None else self.mode
        if kind == 'h5py':
            if self.keep_open:
                if self._f is None or self._f.mode != mode:
                    self._f = h5py.File(self.h5_file, mode)
                return self._f
            return h5py.File(self.h5_file, mode)
        if kind == 'pd':
            if self.keep_open:
                if self._store is None or self._store._mode != mode:
                    self._store = pd.HDFStore(self.h5_file, mode)
                return self._store
            return pd.HDFStore(self.h5_file, mode)


def _load(path, schema={}):
    """
    extract data from a file according to a schema
    """
    out = {key: None for key in schema.keys()}

    for f_name, f in schema.items():
        if not getattr(type(f), 'load', Feature.load) != Feature.load:
            # object doesn't implement load()
            continue
        extensions = getattr(f, '__ext__', {})
        if hasattr(f, 'derived_from') and f.derived_from in out:
            obj = f.load(out[f.derived_from])
        # check that extensions matches
        elif os.path.splitext(path)[-1].strip('.') in extensions or \
                "any" in extensions:
            obj = f.load(path)
        else:
            obj = None
        if not isinstance(obj, (np.ndarray, pd.DataFrame, dict, type(None))):
            raise TypeError(f"cannot write object of type {obj.__class__.__qualname__} to h5mediaset format")
        out[f_name] = obj
    return out

