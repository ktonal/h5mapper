import h5py
import numpy as np
import re
from functools import reduce

__all__ = [
    '_load',
    'apply_and_store',
    '_add',
    'NP_KEY', "SRC_KEY", "REF_KEY",
    'null_regref'
]

H5_NONE = h5py.Empty(np.dtype("S10"))
null_regref = None
NP_KEY = "__arr__"
SRC_KEY = "__src__"
REF_KEY = "__ref__"


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

    for feat_name, feature in schema.items():
        if not getattr(type(feature), 'load', guard_func) != guard_func:
            # object doesn't implement load()
            out.pop(feat_name)
            continue
        regex = getattr(feature, '__re__', r"^\b$")  # default is an impossible to match regex
        if hasattr(feature, 'derived_from') and feature.derived_from in out:
            obj = feature.load(out[feature.derived_from])
        # check that regex matches
        elif re.search(regex, source):
            obj = feature.load(source)
        else:
            obj = None
        if not isinstance(obj, (np.ndarray, dict, type(None))):
            raise TypeError(f"cannot write object of type {obj.__class__.__qualname__} to h5mapper format")
        if obj is None:
            out.pop(feat_name)
        else:
            out[feat_name] = obj
    return out


def apply_and_store(src, fdict, proxy):
    return src, {k: f(proxy.get(src)) for k, f in fdict.items()}


class _add:
    """
    methods to append (create if need be) different types of data to h5Groups/HDFStores

    those methods handle only the logic of adding different types of data to live object
    and do not manage or modify transactions (opening, flushing, closing) in any way.
    Doing so is left to the responsibility of the caller.
    """

    @staticmethod
    def array(group, ds_key, array, ds_kwargs={}):
        if ds_key not in group:
            # kwargs are initialized from first example if need be
            ds_kwargs.setdefault("dtype", array.dtype)
            ds_kwargs.setdefault('shape', (0, *array.shape[1:]))
            ds_kwargs.setdefault('maxshape', (None, *array.shape[1:]))
            ds_kwargs.setdefault('chunks', array.shape)
            group.create_dataset(ds_key, **ds_kwargs)
        # append new data to pre-existing
        offset = group[ds_key].shape[0]
        new = array.shape[0]
        ds = group[ds_key]
        ds.resize((offset + new, *ds.shape[1:]))
        ds[offset:offset + new] = array
        # return a ref to this array
        return ds.regionref[offset:offset + new]

    @staticmethod
    def source_ref(file, path, regionref=None):
        _add.array(file, path + "/" + REF_KEY, np.array([regionref]),
                   dict(dtype=h5py.regionref_dtype))

    @staticmethod
    def data(file, path, data, ds_kwargs):
        if isinstance(data, np.ndarray):
            # if we were to do `_add.array(group, src_name...)` we could
            # create a dataset for each source instead of concatenating the sources immediately...
            ref = _add.array(file, path + "/" + NP_KEY, data, ds_kwargs)
            return ref
        elif isinstance(data, dict):
            # recurse
            refs = {}
            for k in data.keys():
                kwargs = reduce(lambda d, x: d.get(x, {}), (path + k).split('/'), ds_kwargs)
                refs[path + k] = _add.data(file, path + k, data[k], kwargs)
            return refs

    @staticmethod
    def id(file, src, is_new_id=True):
        if SRC_KEY + "/id" not in file or is_new_id:
            _add.array(file, SRC_KEY + "/id", np.array([src]),
                       ds_kwargs=dict(dtype=h5py.string_dtype(encoding='utf-8')))
            return file[SRC_KEY + "/id"].shape[0] - 1
        return (file[SRC_KEY + "/id"].asstr()[:] == src).nonzero()[0][0]

    @staticmethod
    def refs(file, refs, refed, ref_idx):
        # add not yet referenced paths
        for r in (refs.keys() - refed):
            if ref_idx > 0:
                n_refs = file[SRC_KEY + "/id"].shape[0]
                _add.array(file, r + "/" + REF_KEY,
                           np.array([null_regref] * n_refs),
                           dict(dtype=h5py.regionref_dtype))
            _add.array(file, SRC_KEY + "/ds_keys", np.array([r]),
                       dict(dtype=h5py.string_dtype(encoding='utf-8')))
        # add null ref for refed paths not in refs
        for r in (refed - refs.keys()):
            if file[r + "/" + REF_KEY].shape[0]-1 < ref_idx:
                refs.setdefault(r, null_regref)
        # now add the refs
        for path, ref in refs.items():
            if (path + "/" + REF_KEY) not in file or file[path + "/" + REF_KEY].shape[0]-1 < ref_idx:
                _add.source_ref(file, path, ref)
            else:
                # just update the ref
                file[path + "/" + REF_KEY][ref_idx] = ref
        return refs

    @staticmethod
    def source(file, src, data, ds_kwargs, refed, is_new_src=True):
        refs = _add.data(file, "", data, ds_kwargs)
        ref_idx = _add.id(file, src, is_new_src)
        refs = _add.refs(file, refs, refed, ref_idx)
        return refs

