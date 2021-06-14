import h5py
import numpy as np
import pandas as pd
import os


H5_NONE = h5py.Empty(np.dtype("S10"))
with h5py.File("/tmp/__dummy.h5", 'w') as f:
    ds = f.create_dataset('regref', shape=(1, ), dtype=h5py.regionref_dtype)
    null_regref = ds[0]
os.remove("/tmp/__dummy.h5")

DF_KEY = "__df__"
NP_KEY = "__arr__"
PRIVATE_GRP_KEYS = {DF_KEY, NP_KEY}
SRC_KEY = "src"
ID_KEY = "/id"
REF_KEY = "/ref"
KEYS_KEY = "/key"
SRC_ID_KEY = SRC_KEY + ID_KEY
SRC_REF_KEY = SRC_KEY + REF_KEY
SRC_KEYS_KEY = SRC_KEY + KEYS_KEY


class _add:
    """
    methods to append (create if need be) different types of data to h5Groups/HDFStores

    those methods handle only the logic of adding different types of data to live object
    and do not manage or modify transactions (opening, flushing, closing) in any way.
    Doing so is left to the responsibility of the caller.
    """

    @staticmethod
    def array(h5_group: h5py.Group, ds_key, array, ds_kwargs={}):
        if ds_key not in h5_group:
            # kwargs are initialized from first example if need be
            ds_kwargs.setdefault("dtype", array.dtype)
            ds_kwargs.setdefault('shape', (0, *array.shape[1:]))
            ds_kwargs.setdefault('maxshape', (None, *array.shape[1:]))
            ds_kwargs.setdefault('chunks', (1, *array.shape[1:]))
            h5_group.create_dataset(ds_key, **ds_kwargs)
        # append new data to pre-existing
        offset = h5_group[ds_key].shape[0]
        new = array.shape[0]
        ds = h5_group[ds_key]
        ds.resize((offset + new, *array.shape[1:]))
        ds[offset:offset + new] = array
        # return a ref to this array
        return ds.regionref[offset:offset + new]

    @staticmethod
    def source_ref(h5_group, src_name, regionref=None, key=""):
        # print(h5_group, src_name, regionref, key)
        _add.array(h5_group, SRC_ID_KEY, np.array([src_name]),
                   dict(dtype=h5py.string_dtype(encoding='utf-8')))
        _add.array(h5_group, SRC_REF_KEY, np.array([regionref]),
                   dict(dtype=h5py.regionref_dtype))
        if key:
            _add.array(h5_group, SRC_KEYS_KEY, np.array([key]),
                       dict(dtype=h5py.string_dtype(encoding='utf-8')))

    @staticmethod
    def source(src_name, data, group, ds_kwargs, store):
        if isinstance(data, np.ndarray):
            ref = _add.array(group, NP_KEY, data, ds_kwargs)
            _add.source_ref(group, src_name, ref)
            return ref
        elif isinstance(data, pd.DataFrame):
            mi = pd.MultiIndex.from_tuples(zip([src_name] * len(data), data.index), names=["source", ''])
            data = pd.DataFrame(data.reset_index(drop=True).values, index=mi)
            data.to_hdf(store, group.name + "/" + DF_KEY, mode='a', append=True, format="table",
                        # min string size
                        min_itemsize=512)
            return null_regref
        elif isinstance(data, dict):
            # recurse
            for k in data.keys():
                ref = _add.source(src_name, data[k], group.require_group(k),
                                  ds_kwargs.get(k, {}), store)
                _add.source_ref(group, src_name, ref, k)
            return null_regref

    @staticmethod
    def groups(dest, groups):
        def _update(name, obj):
            if isinstance(obj, h5py.Group):
                if obj.name in dest and len(dest[obj.name].keys()) != 0:
                    return None
                # pop in dest only if its empty
                if obj.name in dest:
                    dest.pop(obj.name)
                dest.copy(obj, obj.name)
                return None
            return None
        for grp in groups:
            grp.visititems(_update)
