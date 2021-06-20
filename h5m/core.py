import h5py
import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Optional
from functools import partial, reduce
import tempfile

from .features import Feature
from .crud import _load, _add, DF_KEY, NP_KEY, PRIVATE_GRP_KEYS, SRC_KEY, ID_KEY, REF_KEY, KEYS_KEY, H5_NONE


def as_temp(filename):
    td = tempfile.mktemp()
    os.makedirs(td, exist_ok=True)
    return os.path.join(td, filename)


def in_mem(cls):
    return cls(tempfile.mktemp() + ".h5", "w", keep_open=True, driver='core', backing_store=False)


class Proxy:

    def __getattr__(self, attr):
        if attr in self.__dict__:
            val = self.__dict__[attr]
        # access the feature's methods
        elif self.feature is not None and getattr(self.feature, attr, False):
            val = getattr(self.feature, attr)
        else:
            raise AttributeError(f"object of type {self.__class__.__qualname__} has no attribute '{attr}'")
        return val

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
        self.name = "/".join([group_name.strip("/"), key.strip("/")])
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
            # with a non-disruptive handler (hopefully...)
            was_open = bool(self.owner._f)
            h5f = self.handler("r")
            ds = h5f[self.name]
            self.shape, self.dtype, self.size, self.nbytes = ds.shape, ds.dtype, ds.size, ds.nbytes
            self.ndim, self.maxshape, self.chunks = ds.ndim, ds.maxshape, ds.chunks
            self.asstr = h5py.check_string_dtype(ds.dtype)
            if not was_open:
                h5f.close()

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
            return self._getgrp(item)
        if self.owner.keep_open:
            ds = self.handler()[self.name]
            if getattr(self, "asstr", False):
                ds = ds.asstr()
            rv = ds[item]
            # apply the feature's transform
            if getattr(self, "__t__", tuple()):
                rv = reduce(lambda x, func: func(x), getattr(self, '__t__'), rv)
            return rv
        with self.handler() as f:
            ds = f[self.name]
            if getattr(self, "asstr", False):
                ds = ds.asstr()
            rv = ds[item]
            # apply the feature's transform
            if getattr(self, "__t__", tuple()):
                rv = reduce(lambda x, func: func(x), getattr(self, '__t__'), rv)
        return rv

    def __setitem__(self, item, value):
        if self.is_group:
            return self._setgrp(item, value)
        if self.owner.keep_open:
            self.handler("r+")[self.name][item] = value
        else:
            with self.handler("r+") as f:
                f[self.name][item] = value

    def __repr__(self):
        return f"<Proxy {self.name}>"

    def _getgrp(self, src: str):
        """getitem for groups - here items are the sources' ids"""
        # item is a source name
        if isinstance(src, str):
            # if we were to store the indices of the ids, we wouldn't have
            # to read them all every time...
            mask = self.src.ids[:] == src
            refs, ks = self.src.refs[mask], self.src.keys[mask]
            out = {}
            for ref, k in zip(refs, ks):
                feat = getattr(self, k)
                if ref:  # only arrays have refs
                    o = feat[ref]
                else:
                    if feat.is_group:
                        o = feat[src]
                    else:  # dataframe
                        o = feat[:].loc[src]
                out[k] = o
            return out
        else:
            raise TypeError(f"item should be of type str. Got {type(src)}")

    def _setgrp(self, src: str, value: dict):
        """setitem for groups - here items are the sources' ids"""
        if isinstance(src, str):
            # item = source name
            mask = self.src.ids[:] == src
            if not np.any(mask):
                self.add(src, value)
                return
            refs, ks = self.src.refs[mask], self.src.keys[mask]
            for ref, k in zip(refs, ks):
                feat = getattr(self, k)
                if ref:  # only arrays have refs
                    # will (probably) raise some errors if one's isn't careful with shapes...
                    feat[ref] = value[k]
                else:
                    # feat is either a group or df, but the call is the same
                    feat[src] = value[k]
        else:
            raise TypeError(f"item should be of type str. Got {type(src)}")
        self._attach_new_children(value)

    def _attach_new_children(self, data):
        new_feats = {k for k in data.keys() if not getattr(self, k, False)}
        for new in new_feats:
            key = NP_KEY if isinstance(data[new], np.ndarray) else ""
            setattr(self, new, Proxy(self.owner, self.feature, self.name + new, key))

    def add(self, source, data):
        h5f = self.handler("r+" if self.owner.mode not in ("w", "r+", "a") else self.owner.mode)
        # can't have the 2 handlers writing on 1 file
        # store = self.owner.handler("pd", mode="r+")
        ref = _add.source(h5f[self.group_name],
                          source, data, self.feature.__ds_kwargs__, None)
        h5f.flush()
        if isinstance(data, dict):
            self._attach_new_children(data)
        return ref

    def get(self, source):
        if self.is_group:
            return self._getgrp(source)
        if self.kind == "pd":
            return self[:].loc(source)
        return self[self.refs[self.ids[()] == source][0]]

    def set(self, source, data):
        if self.is_group:
            return self._setgrp(source, data)
        if self.kind == "pd":
            # todo
            raise NotImplementedError
        else:
            self[self.refs[self.ids[()] == source][0]] = data


class Database:

    def __init__(self, h5_file, mode="r", keep_open=False, **h5_kwargs):
        self.h5_file = h5_file
        self.mode = mode
        self.h5_kwargs = h5_kwargs
        self.keep_open = keep_open
        self._f: Optional[h5py.File] = None
        self._store: Optional[pd.HDFStore] = None
        h5f = self.handler("h5py", mode)
        # build the proxies hierarchy from the top level
        for k in h5f.keys():
            # if a key in a file isn't in this class, attach a base Feature
            feature = getattr(type(self), k, setattr(self, k, Feature()))
            self.__dict__[k] = Proxy.from_group(self, feature, h5f[k])

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
                if not self._f or (mode in ("r+", "a", "w") and self._f.mode == "r"):
                    self.close()
                    self._f = h5py.File(self.h5_file, mode, **self.h5_kwargs)
                return self._f
            return h5py.File(self.h5_file, mode, **self.h5_kwargs)
        if kind == 'pd':
            if self.keep_open:
                if self._store is None or mode not in self._store._mode:
                    self._store = pd.HDFStore(self.h5_file, mode)
                return self._store
            return pd.HDFStore(self.h5_file, mode)

    @classmethod
    def load(cls, source, schema={}):
        if not schema:
            schema = {attr: val for attr, val in cls.__dict__.items() if isinstance(val, Feature)}
        return _load(source, schema, guard_func=Feature.load)

    def _attach_new_children(self, data, handle):
        new_feats = {k for k in data.keys() if k not in self.__dict__}
        for new in new_feats:
            feature = getattr(type(self), new, setattr(self, new, Feature()))
            self.__dict__[new] = Proxy.from_group(self, feature, handle[new])

    def add(self, source, data):
        h5f = self.handler('h5py', mode="r+" if self.mode not in ("w", "r+", "a") else self.mode)
        store = None
        kwargs = {k: v.__ds_kwargs__ for k, v in self.__dict__.items() if isinstance(v, Proxy)}
        ref = _add.source(h5f, source, data, kwargs, store)
        self._attach_new_children(data, h5f)
        return ref

    def get(self, source):
        return {k: v.get(source)
                for k, v in self.__dict__.items() if isinstance(v, Proxy)}

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

    def info(self):
        # preserve current handler state
        h5f = h5py.File(self.h5_file, 'r')

        def tostr(name):
            parts = name.split("/")
            s = " " * (len(parts) - 1 + len("/".join(parts[:-1]))) + parts[-1] + "/"
            if isinstance(h5f[name], h5py.Group):
                pass
            else:
                s += "  " + repr(h5f[name].shape) + "  " + repr(h5f[name].dtype)
            print(s)

        h5f.visit(tostr)
        h5f.close()
