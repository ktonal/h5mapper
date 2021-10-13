import h5py
import numpy as np
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import os
from typing import Optional
from functools import reduce
import tempfile

from .create import _create, _compute
from .features import Feature, Array
from .serve import ProgrammableDataset
from .crud import _load, _add, NP_KEY, SRC_KEY, REF_KEY, H5_NONE
from .utils import flatten_dict


__all__ = [
    'as_temp',
    'in_mem',
    'TypedFile',
    'Proxy',
    'typedfile'
]


def as_temp(filename):
    td = tempfile.mktemp()
    os.makedirs(td, exist_ok=True)
    return os.path.join(td, filename)


def in_mem(cls):
    return cls(tempfile.mktemp() + ".h5", "w", keep_open=True, driver='core', backing_store=False)


class Proxy:
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'asstr' in state:
            state.pop('asstr')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            val = self.__dict__[attr]
        elif attr in type(self).__dict__:
            val = type(self).__dict__[attr]
        # access the feature's methods
        elif self.__dict__.get("feature", None) is not None and getattr(self.feature, attr, False):
            val = getattr(self.feature, attr)
        else:
            raise AttributeError(f"object of type {self.__class__.__qualname__} has no attribute '{attr}'")
        return val

    @property
    def h5_(self):
        """returns the h5py object corresponding to this proxy"""
        return self.handle()[self.name]

    @classmethod
    def from_group(cls, owner, feature, group):
        """
        this is the recursive method that build a Proxy hierarchy from a live H5Group
        
        Parameters
        ----------
        owner : TypedFile
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
        attrs = {k: None if not isinstance(v, (np.ndarray, np.void)) and v == H5_NONE else v
                 for k, v in group.attrs.items()}
        # we "lift up" all x/__arr__ to attributes named "x"
        if NP_KEY in group.keys() or isinstance(feature, Array):
            root = Proxy(owner, feature, group.name, "/" + NP_KEY, attrs)
            refs = Proxy(owner, None, group.name, "/" + REF_KEY)
            setattr(root, "refs", refs)
        else:
            root = Proxy(owner, feature, group.name, "", attrs)
        # recursively add children as attributes before returning the object
        root.children = {}
        reserved = dict(REF_KEY='refs', SRC_KEY='src')
        for k in group.keys():
            if isinstance(group[k], h5py.Group):
                child = Proxy.from_group(owner, feature, group[k])
            else:
                attrs = {k: None if v == H5_NONE else v for k, v in group[k].attrs.items()}
                child = Proxy(owner,
                              None if k in reserved else feature,
                              group.name, "/" + k, attrs)
            setattr(root, reserved.get(k, k), child)
            root.children[reserved.get(k, k)] = child
        return root

    def __init__(self, owner, feature, group_name, key="", attrs={}):
        self.name = "/".join([group_name.strip("/"), key.strip("/")])
        self.group_name = group_name
        self.owner: TypedFile = owner
        self.feature = feature
        self.attrs = attrs
        self.is_group = not bool(key)
        if not self.is_group:
            # copy some of the dataset properties
            was_open = bool(self.owner.f_)
            h5f = self.handle("r")
            try:
                ds = h5f[self.name]
                self.shape, self.dtype, self.size, self.nbytes = ds.shape, ds.dtype, ds.size, ds.nbytes
                self.ndim, self.maxshape, self.chunks = ds.ndim, ds.maxshape, ds.chunks
                self.asstr = h5py.check_string_dtype(ds.dtype)
            except KeyError:  # Dataset hasn't been created yet...
                pass
            if not was_open:
                h5f.close()

    def handle(self, mode=None):
        """
        to accommodate torch's DataLoader, h5py.File objects
        are requested in __getitem__ and __setitem__ by proxies, but
        to avoid I/O conflicts, they are instantiated only once by the root TypedFile object
         
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

    def __len__(self):
        # TODO : Should this actually return the number of ids?
        if self.is_group:
            raise TypeError("Group Proxies have no length")
        return self.shape[0]

    def __repr__(self):
        return f"<Proxy {self.name}>"

    def get(self, source):
        if self.is_group:
            return self._getgrp(source)
        if hasattr(self, 'refs'):
            r = self.refs[self.owner.index[source]]
            if r:
                return self[r]
        return None

    def _getgrp(self, src: str):
        if SRC_KEY in self.name:
            return None
        # item is a source name
        if isinstance(src, str):
            out = {}
            for k, proxy in self.children.items():
                rv = proxy.get(src)
                if rv is not None:
                    out[k] = rv
            if getattr(self, "__grp_t__", tuple()):
                out = reduce(lambda x, func: func(x), getattr(self, '__grp_t__'), out)
            return out
        else:
            raise TypeError(f"item should be of type str. Got {type(src)}")

    def set(self, source, data):
        if self.is_group:
            return self._setgrp(source, data)
        else:
            r = self.refs[self.owner.index[source]]
            if r:
                self[r] = data
            else:
                r = _add.array(self.handle('r+'), self.name, data)
                self.refs[self.owner.index[source]] = r

    def _setgrp(self, src: str, value: dict):
        """setitem for groups - here items are the sources' ids"""
        if SRC_KEY in self.name:
            return
        if isinstance(src, str):
            if src not in self.owner.index:
                self.add(src, value)
                return
            # item = source name
            for k, proxy in self.children.items():
                if k in value:
                    proxy.set(src, value[k])
            self._attach_new_children(value)
        else:
            raise TypeError(f"item should be of type str. Got {type(src)}")

    def _attach_new_children(self, data):
        new_feats = {k for k in data.keys() if getattr(self, k, None) is None}
        for new in new_feats:
            key = NP_KEY if isinstance(data[new], np.ndarray) else ""
            proxy = Proxy(self.owner, self.feature, self.name + new, key)
            if key is NP_KEY:
                setattr(proxy, "refs", Proxy(self.owner, None, self.name + new, REF_KEY))
            setattr(self, new, proxy)
            self.children[new] = proxy

    def add(self, source, data):
        h5f = self.handle("r+" if self.owner.mode not in ("w", "r+", "a") else self.owner.mode)
        data_prefixed = flatten_dict(data, prefix=self.group_name.strip("/"))
        _add.source(h5f, source, data_prefixed, self.feature.__ds_kwargs__, self.owner.ds_keys,
                    source not in self.owner.index)
        self.owner.build_proxies()
        if isinstance(data, dict):
            self._attach_new_children(data)
        return self

    def iset(self, source, idx, data):
        if self.is_group:
            return self._setgrp(source, data)
        else:
            ds = self.handle()[self.name]
            ref = self.refs[self.owner.index[source]]
            # get the MultiBlockSlice behind the regionref :
            mbs = h5py.h5r.get_region(ref, ds.id).get_regular_hyperslab()
            # mbs[0][0] is the first index of the first axis of the region
            ds[mbs[0][0] + idx] = data

    def close(self):
        return self.owner.close()

    def flush(self):
        return self.owner.flush()

    def compute(self, fdict, n_workers=cpu_count(), parallelism='mp', destination=None):
        """
        apply `fdict` to each of self's sources and store the result in `destination`
        """
        _compute(fdict, self, parallelism, n_workers,
                 self.owner if destination is None else destination)
        return destination

    def serve(self, batch, **loader_kwargs):
        return self.owner.serve(batch, **loader_kwargs)

    @property
    def sources(self):
        return getattr(self.owner, SRC_KEY).id[self.refs[:].astype(np.bool)]


class TypedFile:

    def __init__(self, filename, mode="r", keep_open=False, **h5_kwargs):
        self.ds_keys = set()
        self.index = {}
        self.filename = filename
        self.mode = mode
        self.h5_kwargs = h5_kwargs
        self.keep_open = keep_open
        self.f_: Optional[h5py.File] = None
        self.init_schema()
        self.build_proxies()
        # we reset this after init for being able to pickle this object
        # in multiprocessing contexts
        self.f_ = None

    def init_schema(self):
        if self.mode == 'r':
            self.build_proxies()
            return self
        h5 = self.handle()
        for attr, val in type(self).__dict__.items():
            if isinstance(val, Feature):
                h5.require_group(attr)
        h5.require_group(SRC_KEY)
        self.build_proxies()
        return self

    def build_proxies(self):
        h5f = self.handle()
        for k, val in type(self).__dict__.items():
            if isinstance(val, Feature):
                self.__dict__[k] = Proxy(self, val, k)
        # build the proxies hierarchy from the top level
        for k in h5f.keys():
            # if a key in a file isn't in this class, attach a base Feature
            feature = getattr(type(self), k, None)
            if feature is None:
                feature = Feature()
                setattr(self, k, feature)
            if isinstance(h5f[k], h5py.Group):
                before = self.__dict__[k]
                self.__dict__[k] = Proxy.from_group(self, feature, h5f[k])
                assert self.__dict__[k] is not before
            else:
                self.__dict__[k] = Proxy(self, feature, h5f[k].parent.name, k, dict(h5f[k].attrs))
        self.build_refs()

    def build_refs(self):
        if hasattr(self, SRC_KEY):
            ids = getattr(getattr(self, SRC_KEY), "id", [])[:]
            self.index = dict(zip(ids, range(len(ids))))
            ds_keys = getattr(getattr(self, SRC_KEY), "ds_keys", [])[:]
            self.ds_keys = set(ds_keys)

    @classmethod
    def create(cls,
               filename,
               sources,
               mode="w",
               schema={},
               n_workers=cpu_count(),
               parallelism='mp',
               keep_open=False,
               **h5_kwargs
               ):
        return _create(cls,
                       filename,
                       sources,
                       mode,
                       schema,
                       n_workers,
                       parallelism,
                       keep_open,
                       **h5_kwargs)

    def handle(self, mode=None):
        """
        """
        mode = mode if mode is not None else self.mode
        if self.keep_open:
            if not self.f_ or (mode in ("r+", "a", "w") and self.f_.mode == "r"):
                self.close()
                self.f_ = h5py.File(self.filename, mode, **self.h5_kwargs)
            return self.f_
        h = h5py.File(self.filename, mode, **self.h5_kwargs)
        if mode == self.mode == 'w':
            self.mode = 'r+'
        return h

    @classmethod
    def load(cls, source, schema={}):
        if not schema:
            schema = {attr: val for attr, val in cls.__dict__.items() if isinstance(val, Feature)}
        return _load(source, schema, guard_func=Feature.load)

    def _attach_new_children(self, data, handle):
        for new in data.keys():
            feature = getattr(type(self), new, setattr(self, new, Feature()))
            proxy = Proxy.from_group(self, feature, handle[new])
            self.__dict__[new] = proxy

    def add(self, source, data):
        h5f = self.handle(mode="r+" if self.mode not in ("w", "r+", "a") else self.mode)
        kwargs = {k: getattr(v, "__ds_kwargs__", {}) for k, v in self.__dict__.items() if isinstance(v, Proxy)}
        data = flatten_dict(data)
        _add.source(h5f, source, data, kwargs, self.ds_keys, source not in self.index)
        self.build_proxies()
        return self

    def get(self, source):
        return {k: v.get(source)
                for k, v in self.__dict__.items() if isinstance(v, Proxy)}

    def set(self, source, val):
        pass

    def get_proxy(self, name):
        """get the proxy for an h5 name, e.g. ``'net/fc.weight'``"""
        name = name.strip("/").split('/')
        return reduce(getattr, name, self)

    def flush(self):
        if self.f_:
            self.f_.flush()

    def close(self):
        if getattr(self, 'f_', None) is not None:
            self.f_.close()
            if self.mode == "w":
                # reopening a created doesn't overwrite it
                self.mode = 'r+'
        return

    def open(self, mode):
        self.mode = mode
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        self.close()

    def __del__(self):
        self.close()

    def __repr__(self):
        return f"<{type(self).__qualname__} {self.filename}>"

    def info(self):
        # preserve current handler state
        h5f = self.handle("r")

        def tostr(name):
            parts = name.split("/")
            s = " " * (len(parts) - 1 + len("/".join(parts[:-1]))) + parts[-1] + "/"
            if isinstance(h5f[name], h5py.Group):
                pass
            else:
                s += "  " + repr(h5f[name].shape) + "  " + repr(h5f[name].dtype)
            print(s)

        print("---- ", self, " ----")
        h5f.visit(tostr)
        h5f.close()

    def serve(self, batch, **loader_kwargs):
        ds = ProgrammableDataset(self, batch)
        return DataLoader(ds, **loader_kwargs)


def typedfile(name, schema):
    return type(name, (TypedFile,), schema)


# TODO : SingleID object ~ TypedFile with only one ref (copy? mask? link?...) for concatenating datasets

