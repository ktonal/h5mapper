import h5py
import numpy as np
import os
import re

H5_NONE = h5py.Empty(np.dtype("S10"))
with h5py.File("/tmp/__dummy.h5", 'w') as f:
    ds = f.create_dataset('regref', shape=(1,), dtype=h5py.regionref_dtype)
    null_regref = ds[0]
os.remove("/tmp/__dummy.h5")

NP_KEY = "__arr__"
SRC_KEY = "src"
PRIVATE_GRP_KEYS = {NP_KEY}
ID_KEY = "/ids"
REF_KEY = "/refs"
KEYS_KEY = "/keys"
SRC_ID_KEY = SRC_KEY + ID_KEY
SRC_REF_KEY = SRC_KEY + REF_KEY
SRC_KEYS_KEY = SRC_KEY + KEYS_KEY


def _load(source, schema={}, guard_func=None):
    """
    extract data from a source according to a schema.

    Features whose `__re__` attribute matches against `source` contribute to the
    returned value.

    If a Array has a string stored in an attribute `derived_from`, the single argument
    passed to this Array's load() method changes to being what the "source feature" returned.
    In pseudo-code :
        ```
        if feature.derived_from in out.keys():
            out[feature.name] = feature.load(out[feature.derived_from])
        ```
    Thus, the order of the schema matters and can be used to specify a graph of
    loading functions.

    Parameters
    ----------
    source : str
        the input to be loaded (the path to a file, an url, ...)
    schema : dict
        must have strings as keys (names of the H5 objects) and Features as values
    guard_func : optional callable
        Array whose load is equal to ``guard_func`` are by-passed.
        Typically, `guard_func` is the method of an abstract base class.

    Returns
    -------
    data : dict
        same keys as `schema`, values are the arrays, dataframes or dict returned by the Features
    """
    out = {key: None for key in schema.keys()}

    for f_name, f in schema.items():
        if not getattr(type(f), 'load', guard_func) != guard_func:
            # object doesn't implement load()
            out.pop(f_name)
            continue
        regex = getattr(f, '__re__', r"^\b$")  # default is an impossible to match regex
        if hasattr(f, 'derived_from') and f.derived_from in out:
            obj = f.load(out[f.derived_from])
        # check that regex matches
        elif re.search(regex, source):
            obj = f.load(source)
        else:
            obj = None
        if not isinstance(obj, (np.ndarray, dict, type(None))):
            raise TypeError(f"cannot write object of type {obj.__class__.__qualname__} to h5mapper format")
        out[f_name] = obj
    return out


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
            ds_kwargs.setdefault('chunks', array.shape)
            h5_group.create_dataset(ds_key, **ds_kwargs)
        # append new data to pre-existing
        offset = h5_group[ds_key].shape[0]
        new = array.shape[0]
        ds = h5_group[ds_key]
        ds.resize((offset + new, *ds.shape[1:]))
        ds[offset:offset + new] = array
        # return a ref to this array
        return ds.regionref[offset:offset + new]

    @staticmethod
    def source_ref(h5_group, src_name, regionref=None, key=""):
        """populate the ref & key datasets of a group"""
        # _add.array(h5_group, SRC_ID_KEY, np.array([src_name]),
        #            dict(dtype=h5py.string_dtype(encoding='utf-8')))
        _add.array(h5_group, SRC_REF_KEY, np.array([regionref]),
                   dict(dtype=h5py.regionref_dtype))
        if key:
            _add.array(h5_group, SRC_KEYS_KEY, np.array([key]),
                       dict(dtype=h5py.string_dtype(encoding='utf-8')))
        # group/src.attrs hold the set of ids mapped to the lists of indices in
        # group/src/[refs, keys] they correspond to.
        pkeys = h5_group[SRC_KEY].attrs
        pkeys.create(src_name, [*pkeys.get(src_name, []), h5_group[SRC_REF_KEY].shape[0]-1])

    @staticmethod
    def source(group, src_name, data, ds_kwargs, ):
        if isinstance(data, np.ndarray):
            # if we were to do `_add.array(group, src_name...)` we could
            # create a dataset for each source instead of concatenating the sources immediately...
            ref = _add.array(group, NP_KEY, data, ds_kwargs)
            _add.source_ref(group, src_name, ref)
            return ref
        elif isinstance(data, dict):
            # recurse
            for k in data.keys():
                ref = _add.source(group.require_group(k), src_name, data[k],
                                  ds_kwargs.get(k, {}))
                _add.source_ref(group, src_name, ref, k)
            return null_regref

    # @staticmethod
    # def groups(dest, groups, soft=True):
    #     def _update(name, grp):
    #         if isinstance(grp, h5py.Group):
    #             # if dest has a non-empty group for this key, pass (raise Exception?)
    #             if soft and grp.name in dest and len(dest[grp.name].keys()) != 0:
    #                 return None
    #             # if its empty in dest or soft=False, pop in dest
    #             if grp.name in dest:
    #                 dest.pop(grp.name)
    #             # add this group to dest
    #             dest.copy(grp, grp.name)
    #             return None
    #         return None
    #
    #     for grp in groups:
    #         grp.visititems(_update)

    # @staticmethod
    # def virtual_dataset(group: h5py.Group, vds_key, real_ds_keys):
    #     """concatenate children of a group into a virtual dataset"""
    #     shapes = np.array([group[k].shape for k in real_ds_keys])
    #     assert np.all(shapes[:, 1:] == shapes[0:1, 1:])
    #     layout = h5py.VirtualLayout(shape=(shapes.T[0].sum(), *shapes[0, 1:]),
    #                                 dtype=group[next(iter(real_ds_keys))].dtype)
    #     offset = 0
    #     for k in real_ds_keys:
    #         ds = group[k]
    #         n = ds.shape[0]
    #         vsource = h5py.VirtualSource(group.file, ds.name, ds.shape)
    #         layout[offset:offset + n] = vsource
    #         offset += n
    #     group.create_virtual_dataset(vds_key, layout)
