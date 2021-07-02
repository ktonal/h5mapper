import h5py
from multiprocessing import cpu_count, Pool
from concurrent.futures import ThreadPoolExecutor
import os
from functools import partial

from tqdm import tqdm

from .features import Array
from .crud import _add, _load, H5_NONE, SRC_KEY
from .utils import flatten_dict


__all__ = [
    "_create"
]


def _create(cls,
            filename,
            sources,
            mode="w",
            schema={},
            n_workers=cpu_count(),
            parallelism='mp',
            keep_open=False,
            **h5_kwargs
            ):
    if not schema:
        # get schema from the class attributes
        schema = {attr: val for attr, val in cls.__dict__.items() if isinstance(val, Array)}
    if not schema:
        raise ValueError("schema cannot be empty. Either provide one to create()"
                         " or attach Array attributes to this class.")
    # create two separate files for arrays and dataframes
    f = h5py.File(filename, mode, **h5_kwargs)
    f.require_group(SRC_KEY)
    # create groups from schema and write attrs
    groups = {key: f.create_group(key) if key not in f else f[key] for key in schema.keys()}
    for key, grp in groups.items():
        for k, v in schema[key].attrs.items():
            grp.attrs[k] = v if v is not None else H5_NONE
    f.flush()

    # initialize ds_kwargs from schema
    ds_kwargs = {key: getattr(feature, "__ds_kwargs__", {}).copy() for key, feature in schema.items()}
    # get flavour of parallelism
    if parallelism == 'mp':
        executor = Pool(n_workers)
    elif parallelism == 'future':
        executor = ThreadPoolExecutor(n_workers)
    elif parallelism == 'none':
        class Executor:
            def map(self, func, iterable):
                return map(func, iterable)
        executor = Executor()
    else:
        f.close()
        raise ValueError(f"parallelism must be one of ['mp', 'future']. Got '{parallelism}'")

    # run loading routine
    n_sources = len(sources)
    batch_size = n_workers * 4
    refed_paths = set()
    for i in tqdm(range(1 + n_sources // batch_size), leave=False):
        start_loc = max([i * batch_size, 0])
        end_loc = min([(i + 1) * batch_size, n_sources])
        this_sources = sources[start_loc:end_loc]
        try:
            results = executor.map(partial(_load, schema=schema, guard_func=Array.load), this_sources)
        except Exception as e:
            f.close()
            if mode == "w":
                os.remove(filename)
            if parallelism == 'mp':
                executor.terminate()
            raise e
        # write results
        for n, res in enumerate(results):
            if not res:
                continue
            res = flatten_dict(res)
            _add.source(f, this_sources[n], res, ds_kwargs, refed_paths)
            refed_paths = refed_paths | set(res.keys())
        f.flush()
    if parallelism == 'mp':
        executor.close()
        executor.join()
    # run after_create
    db = cls(filename, mode="r+", keep_open=False)
    for key, feature in schema.items():
        if getattr(type(feature), "after_create", Array.after_create) != Array.after_create:
            feature.after_create(db, key)
            f.flush()
    # voila!
    return cls(filename, mode if mode != 'w' else "r+", keep_open)
