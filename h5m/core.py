import h5py
import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool
from concurrent.futures import ThreadPoolExecutor
import os
import re
from typing import Optional
from functools import partial, reduce
import inspect

from .crud import _add, DF_KEY, NP_KEY, PRIVATE_GRP_KEYS, SRC_KEY, ID_KEY, REF_KEY, KEYS_KEY, H5_NONE


class Feature:
    __re__ = r".*"
    __ds_kwargs__ = {}
    __primary__ = 'src'

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
        return dict(list(item for f in self.features.values() for item in f.attrs.items()))

    def load(self, source):
        return Database.load(source, self.features)

    def after_create(self, db, feature_key):
        for feat in self.features.values():
            if getattr(type(feat), "after_create", Feature.after_create) != Feature.after_create:
                feat.after_create(db, feature_key + "/" + feat.name)


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
        """
        this is the recursive method that build a Proxy hierarchy (attributes) from a live H5Group
        
        Parameters
        ----------
        owner : Database
            the root object holding this proxy
        feature : Feature
            the feature that loaded this group
        group : h5py.H5Group
            the group corresponding to this proxy
            
        Returns
        -------
        proxy : Proxy
            a proxy to `group` with the children attached as attributes
        """
        # print("INSTANTIATING", group.name)
        attrs = {k: None if v == H5_NONE else v for k, v in group.attrs.items()}
        # we "lift up" x/__arr__ or x/__df__ to attributes named "x"
        if NP_KEY in group.keys():
            root = Proxy(owner, feature, group.name, NP_KEY, attrs)
        elif DF_KEY in group.keys():
            root = Proxy(owner, feature, group.name, DF_KEY, attrs)
        else:
            root = Proxy(owner, feature, group.name, "", attrs)
        if SRC_KEY in group.keys():
            refs = Proxy(owner, feature, group.name + "/" + SRC_KEY, REF_KEY)
            ids = Proxy(owner, feature, group.name + "/" + SRC_KEY, ID_KEY)
            # print("ATTACHING", "refs, ids", "TO", root)
            setattr(root, "refs", refs)
            setattr(root, "ids", ids)
            if KEYS_KEY.strip("/") in group[SRC_KEY].keys():
                keys = Proxy(owner, feature, group.name + "/" + SRC_KEY, KEYS_KEY)
                setattr(root, "keys", keys)
                # print("ATTACHING", "keys", "TO", root)
        # dataframes groups do not host their children
        if DF_KEY in group.keys():
            return root
        # recursively add children as attributes before returning the object
        for k in group.keys():
            if isinstance(group[k], h5py.Group) and k not in PRIVATE_GRP_KEYS:
                child = Proxy.from_group(owner, feature, group[k])
            else:
                attrs = {k: None if v == H5_NONE else v for k, v in group[k].attrs.items()}
                child = Proxy(owner, feature, group.name, "/"+k, attrs)
            # print("ATTACHING", k, child, "TO", root)
            setattr(root, k, child)
        return root

    def __init__(self, owner, feature, group_name, key="", attrs={}):
        self.h5_file = owner.h5_file
        self.name = "/".join([group_name, key.strip("/")])
        self.group_name = group_name
        self.owner = owner
        self.feature = feature
        self.attrs = attrs
        self.is_group = not bool(key)
        if key.strip("/") == DF_KEY:
            self.kind = 'pd'
        else:
            self.kind = 'h5py'
        if self.kind == "h5py" and not self.is_group:
            # copy some of the dataset properties
            with self.handler() as h5f:
                ds = h5f[self.name]
                self.shape, self.dtype, self.size, self.nbytes = ds.shape, ds.dtype, ds.size, ds.nbytes
                self.ndim, self.maxshape, self.chunks = ds.ndim, ds.maxshape, ds.chunks
                self.asstr = h5py.check_string_dtype(ds.dtype)

    def handler(self, mode="r"):
        """
        to accommodate torch's DataLoader, handles to .h5 (pd.HDFStore or h5py.File)
        are requested in __getitem__ and __setitem__ by proxies, but
        to avoid I/O conflicts, they are instantiated only once by the root Database object
         
        Returns
        -------
        handle : h5py.File or pd.HDFStore
            a handle that can read/write the data for this Proxy
        """
        return self.owner.handler(self.kind, mode)
    
    def __getitem__(self, item):
        if self.is_group:
            return self.getgrp(item)
        if self.owner.keep_open:
            ds = self.handler()[self.name]
            if getattr(self, "asstr", False):
                ds = ds.asstr()
            return ds[item]
        with self.handler() as f:
            ds = f[self.name]
            if getattr(self, "asstr", False):
                ds = ds.asstr()
            rv = ds[item]
        return rv

    def __setitem__(self, item, value):
        if self.owner.keep_open:
            self.handler()[self.name][item] = value
        else:
            with self.handler() as f:
                f[self.name][item] = value

    def __repr__(self):
        return f"<Proxy {self.name}>"

    def getgrp(self, item):
        if isinstance(item, str):
            # item is a source name
            mask = self.src.ids[:] == item
            refs, ks = self.src.refs[mask], self.src.keys[mask]
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
        else:
            raise TypeError(f"item should be of type str. Got {type(item)}")

    def add(self, source, data):
        h5f = self.owner.handler('h5py', mode="r+")
        store = self.owner.handler("pd", mode="r+")
        ref = _add.source(h5f[self.group_name],
                          source, data, self.feature.__ds_kwargs__, store)
        return ref

    def get(self, source):
        if self.is_group:
            return self.getgrp(source)
        if self.kind == "pd":
            return self[:].loc(source)
        return self[self.refs[self.ids[()] == source]]


class Database:

    def __init__(self, h5_file, mode="r", keep_open=False, **h5_kwargs):
        self.h5_file = h5_file
        self.mode = mode
        self.h5_kwargs = h5_kwargs
        self.keep_open = keep_open
        self._f: Optional[h5py.File] = None
        self._store: Optional[pd.HDFStore] = None
        with self.handler('h5py', 'r') as f:
            # build the proxies hierarchy from the top level
            for k in f.keys():
                # if a key in a file isn't in this class, attach a base Feature
                feature = getattr(type(self), k, setattr(self, k, Feature()))
                self.__dict__[k] = Proxy.from_group(self, feature, f[k])

    @classmethod
    def create(cls,
               h5_file,
               sources,
               mode="w",
               schema={},
               n_workers=cpu_count() * 2,
               parallelism='mp',
               keep_open=False,
               **h5_kwargs
               ):
        if not schema:
            # get schema from the class attributes
            schema = {attr: val for attr, val in cls.__dict__.items() if isinstance(val, Feature)}
        if not schema:
            raise ValueError("schema cannot be empty. Either provide one to create()"
                             " or attach Feature attributes to this class.")
        # create two separate files for arrays and dataframes
        f = h5py.File(h5_file, mode, **h5_kwargs)
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
            # we use Database.load instead of cls.load because cls might be in a <locals>
            # which Pool can not pickle...
            try:
                results = executor.map(partial(Database.load, schema=schema), this_sources)
            except Exception as e:
                f.flush()
                f.close()
                store.close()
                os.remove(h5_file)
                os.remove("/tmp/h5m-store.h5")
                raise e
            # write results
            for n, res in enumerate(results):
                for key, data in res.items():
                    if data is None:
                        continue
                    _add.source(groups[key], this_sources[n], data, ds_kwargs[key], store)
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
                if not self._f or self._f.mode != mode:
                    self._f = h5py.File(self.h5_file, mode, **self.h5_kwargs)
                return self._f
            return h5py.File(self.h5_file, mode, **self.h5_kwargs)
        if kind == 'pd':
            if self.keep_open:
                if self._store is None or self._store._mode != mode:
                    self._store = pd.HDFStore(self.h5_file, mode)
                return self._store
            return pd.HDFStore(self.h5_file, mode)

    @classmethod
    def load(cls, source, schema={}):
        """
        extract data from a source according to a schema.
        
        Only Features whose `__re__` attribute matches against `source` contribute to the 
        returned value.
        
        Parameters
        ----------
        source : str
            
        schema : dict
            must have strings as keys (names of the H5 objects) and Features as values
            
        Returns
        -------
        data : dict
            same keys as `schema`, values are the arrays, dataframes or dict returned by the Features
        """
        if not schema:
            schema = {attr: val for attr, val in cls.__dict__.items() if isinstance(val, Feature)}

        out = {key: None for key in schema.keys()}
    
        for f_name, f in schema.items():
            if not getattr(type(f), 'load', Feature.load) != Feature.load:
                # object doesn't implement load()
                out.pop(f_name)
                continue
            regex = getattr(f, '__re__', r"^\b$")  # default is an impossible to match regex
            if hasattr(f, 'derived_from') and f.derived_from in out:
                obj = f.load(out[f.derived_from])
            # check that regex matches
            elif re.match(regex, source):
                obj = f.load(source)
            else:
                obj = None
            if not isinstance(obj, (np.ndarray, pd.DataFrame, dict, type(None))):
                raise TypeError(f"cannot write object of type {obj.__class__.__qualname__} to h5mapper format")
            out[f_name] = obj
        return out

    def get_feat(self, name):
        name = name.strip("/").split('/')
        return reduce(getattr, name, self)

    def flush(self):
        if self._f:
            self._f.flush()
        if self._store is not None and self._store.is_open:
            self._store.flush()

    def close(self):
        if self._f is not None:
            self._f.close()
        if self._store is not None:
            self._store.close()

    def __del__(self):
        self.close()

    def __repr__(self):
        return f"<Database {self.h5_file}>"
