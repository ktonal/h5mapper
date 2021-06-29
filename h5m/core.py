import h5py
import numpy as np
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import os
from typing import Optional
from functools import reduce
import tempfile

from .create import _create
from .features import Array
from .serve import DefaultDataset
from .crud import _load, _add, NP_KEY, PRIVATE_GRP_KEYS, SRC_KEY, ID_KEY, REF_KEY, KEYS_KEY, H5_NONE


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

    @property
    def h5_(self):
        """returns the h5py object of this proxy"""
        return self.handle()[self.name]

    @property
    def attrs_(self):
        """returns the h5py .attrs of this proxy"""
        return self.handle()[self.group_name].attrs

    @classmethod
    def _from_group(cls, owner, feature, group):
        """
        this is the recursive method that build a Proxy hierarchy from a live H5Group
        
        Parameters
        ----------
        owner : FileType
            the root object holding this proxy
        feature : Array
            the feature that loaded this group
        group : h5py.H5Group
            the group corresponding to this proxy
            
        Returns
        -------
        proxy : Proxy
            a proxy to `group` with the children attached as attributes
        """
        if SRC_KEY not in group.name:
            attrs = {k: None if not isinstance(v, np.ndarray) and v == H5_NONE else v
                     for k, v in group.attrs.items()}
        else:
            attrs = {}
        # we "lift up" x/__arr__ or x/__df__ to attributes named "x"
        if NP_KEY in group.keys():
            root = Proxy(owner, feature, group.name, NP_KEY, attrs)
        else:
            root = Proxy(owner, feature, group.name, "", attrs)
        if SRC_KEY in group.keys():
            refs = Proxy(owner, None, group.name + "/" + SRC_KEY, REF_KEY)
            # ids = Proxy(owner, None, group.name + "/" + SRC_KEY, ID_KEY)
            ids = list(group[SRC_KEY].attrs.keys())
            # print("ATTACHING", "refs, ids", "TO", root)
            setattr(root, "refs", refs)
            setattr(root, "ids", ids)
            if KEYS_KEY.strip("/") in group[SRC_KEY].keys():
                keys = Proxy(owner, None, group.name + "/" + SRC_KEY, KEYS_KEY)
                setattr(root, "keys", keys)
        # recursively add children as attributes before returning the object
        for k in group.keys():
            if isinstance(group[k], h5py.Group) and k not in PRIVATE_GRP_KEYS:
                child = Proxy._from_group(owner, feature, group[k])
            else:
                attrs = {k: None if v == H5_NONE else v for k, v in group[k].attrs.items()}
                child = Proxy(owner,
                              None if any(k in reserved
                                          for reserved in [REF_KEY, ID_KEY, KEYS_KEY])
                              else feature,
                              group.name, "/" + k, attrs)
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
        if not self.is_group:
            # copy some of the dataset properties
            # with a non-disruptive handler (hopefully...)
            was_open = bool(self.owner._f)
            h5f = self.handle("r")
            ds = h5f[self.name]
            self.shape, self.dtype, self.size, self.nbytes = ds.shape, ds.dtype, ds.size, ds.nbytes
            self.ndim, self.maxshape, self.chunks = ds.ndim, ds.maxshape, ds.chunks
            self.asstr = h5py.check_string_dtype(ds.dtype)
            if not was_open:
                h5f.close()

    def handle(self, mode="r"):
        """
        to accommodate torch's DataLoader, h5py.File objects
        are requested in __getitem__ and __setitem__ by proxies, but
        to avoid I/O conflicts, they are instantiated only once by the root FileType object
         
        Returns
        -------
        handle : h5py.File
            object that can read/write the data for this Proxy
        """
        return self.owner.handle(mode)

    def __getitem__(self, item):
        if self.is_group:
            return self._getgrp(item)
        if self.owner.keep_open:
            ds = self.handle()[self.name]
            if getattr(self, "asstr", False):
                ds = ds.asstr()
            rv = ds[item]
            # apply the feature's transform
            if getattr(self, "__t__", tuple()):
                rv = reduce(lambda x, func: func(x), getattr(self, '__t__'), rv)
            return rv
        with self.handle() as f:
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
            self.handle("r+")[self.name][item] = value
        else:
            with self.handle("r+") as f:
                f[self.name][item] = value

    def __repr__(self):
        return f"<Proxy {self.name}>"

    def _getgrp(self, src: str):
        """getitem for groups - here items are the sources' ids"""
        # item is a source name
        if isinstance(src, str):
            if not hasattr(self, "src"):
                # not a h5m Group
                return {}
            # we store the map ids <-> indices for refs and keys
            # in the src.attrs of the feature
            mask = self.src.attrs_[src]
            refs, ks = self.src.refs[mask], self.src.keys[mask]
            out = {}
            for ref, k in zip(refs, ks):
                feat = getattr(self, k)
                if ref:  # only arrays have refs
                    o = feat[ref]
                else:
                    o = feat[src]
                out[k] = o
            if getattr(self, "__grp_t__", tuple()):
                out = reduce(lambda x, func: func(x), getattr(self, '__grp_t__'), out)
            return out
        else:
            raise TypeError(f"item should be of type str. Got {type(src)}")

    def _setgrp(self, src: str, value: dict):
        """setitem for groups - here items are the sources' ids"""
        if isinstance(src, str):
            if not hasattr(self, "src"):
                print("no SRC")
                # not a h5m Group
                return
            # item = source name
            mask = self.src.attrs_.get(src, [])
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
                    # feat is a group
                    feat[src] = value[k]
            self._attach_new_children(value)
        else:
            raise TypeError(f"item should be of type str. Got {type(src)}")

    def _attach_new_children(self, data):
        new_feats = {k for k in data.keys() if not getattr(self, k, False)}
        for new in new_feats:
            key = NP_KEY if isinstance(data[new], np.ndarray) else ""
            setattr(self, new, Proxy(self.owner, self.feature, self.name + new, key))

    def add(self, source, data):
        h5f = self.handle("r+" if self.owner.mode not in ("w", "r+", "a") else self.owner.mode)
        ref = _add.source(h5f[self.group_name],
                          source, data, self.feature.__ds_kwargs__)
        # h5f.flush()
        if isinstance(data, dict):
            self._attach_new_children(data)
        return ref

    def get(self, source):
        if self.is_group:
            return self._getgrp(source)
        return self[self.refs[self.src.attrs_[source][0]]]

    def set(self, source, data):
        if self.is_group:
            return self._setgrp(source, data)
        else:
            self[self.refs[self.src.attrs_[source][0]]] = data

    def iset(self, source, idx, data):
        if self.is_group:
            # print("ISET GRP")
            return self._setgrp(source, data)
        else:
            ds = self.handle()[self.name]
            ref = self.refs[self.src.attrs_[source][0]]
            # get the MultiBlockSlice behind the regionref :
            mbs = h5py.h5r.get_region(ref, ds.id).get_regular_hyperslab()
            # mbs[0][0] is the first index of the first axis of the region
            ds[mbs[0][0] + idx] = data


class FileType:

    def __init__(self, h5_file, mode="r", keep_open=False, **h5_kwargs):
        self.h5_file = h5_file
        self.mode = mode
        self.h5_kwargs = h5_kwargs
        self.keep_open = keep_open
        self._f: Optional[h5py.File] = None
        h5f = self.handle(mode)
        for k, val in type(self).__dict__.items():
            if isinstance(val, Array):
                self.__dict__[k] = Proxy(self, val, k)
        # build the proxies hierarchy from the top level
        for k in h5f.keys():
            # if a key in a file isn't in this class, attach a base Array
            feature = getattr(type(self), k, setattr(self, k, Array()))
            self.__dict__[k] = Proxy._from_group(self, feature, h5f[k])

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
        return _create(cls, h5_file, sources, mode, schema, n_workers, parallelism, keep_open, **h5_kwargs)

    def handle(self, mode=None):
        """
        """
        mode = mode if mode is not None else self.mode
        if self.keep_open:
            if not self._f or (mode in ("r+", "a", "w") and self._f.mode == "r"):
                self.close()
                self._f = h5py.File(self.h5_file, mode, **self.h5_kwargs)
            return self._f
        return h5py.File(self.h5_file, mode, **self.h5_kwargs)

    @classmethod
    def load(cls, source, schema={}):
        if not schema:
            schema = {attr: val for attr, val in cls.__dict__.items() if isinstance(val, Array)}
        return _load(source, schema, guard_func=Array.load)

    def _attach_new_children(self, data, handle):
        # new_feats = {k for k in data.keys() if k not in self.__dict__}
        for new in data.keys():
            feature = getattr(type(self), new, setattr(self, new, Array()))
            proxy = Proxy._from_group(self, feature, handle[new])
            self.__dict__[new] = proxy

    def add(self, source, data):
        h5f = self.handle(mode="r+" if self.mode not in ("w", "r+", "a") else self.mode)
        kwargs = {k: getattr(v, "__ds_kwargs__", {}) for k, v in self.__dict__.items() if isinstance(v, Proxy)}
        ref = _add.source(h5f, source, data, kwargs)
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

    def close(self):
        if self._f:
            self._f.close()

    def __del__(self):
        self.close()

    def __repr__(self):
        return f"<FileType {self.h5_file}>"

    def info(self):
        # preserve current handler state
        h5f = self.handle("r")

        def tostr(name):
            parts = name.split("/")
            s = " " * (len(parts) - 1 + len("/".join(parts[:-1]))) + parts[-1] + "/"
            if isinstance(h5f[name], h5py.Group):
                if SRC_KEY in name:
                    s += f"  ({len(h5f[name].attrs.keys())} ids)"
            else:
                s += "  " + repr(h5f[name].shape) + "  " + repr(h5f[name].dtype)
            print(s)

        print("---- ", self, " ----")
        h5f.visit(tostr)
        h5f.close()

    def serve(self, batch, **loader_kwargs):
        ds = DefaultDataset(self, batch)
        return DataLoader(ds, **loader_kwargs)


def filetype(name, schema):
    return type(name, (FileType, ), schema)
