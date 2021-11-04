import h5py
import numpy as np
from multiprocess import cpu_count, Pool
from concurrent.futures import ThreadPoolExecutor
import os
from functools import partial

from tqdm import tqdm

from .features import Feature
from .crud import _add, _load, H5_NONE, SRC_KEY, apply_and_store
from .utils import flatten_dict


__all__ = [
    "_create",
    '_compute',
]


class SerialExecutor:
    def map(self, func, iterable):
        return map(func, iterable)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def close(self):
        pass

    def join(self):
        pass


def get_executor(n_workers=cpu_count(), parallelism='mp'):
    if parallelism == 'mp':
        executor = Pool(n_workers)
    elif parallelism == 'threads':
        executor = ThreadPoolExecutor(n_workers)
    elif parallelism == 'none':
        executor = SerialExecutor()
    else:
        raise ValueError(f"parallelism must be one of ['mp', 'threads', 'none']. Got '{parallelism}'")
    return executor


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
        schema = {attr: val for attr, val in cls.__dict__.items() if isinstance(val, Feature)}
    if not schema:
        raise ValueError("schema cannot be empty. Either provide one to create()"
                         " or attach Feature attributes to this class.")
    # avoid blocking errors from h5py
    if os.path.exists(filename) and mode == 'w':
        os.remove(filename)
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
    try:
        executor = get_executor(n_workers, parallelism)
    except ValueError as e:
        f.close()
        raise e
    # run loading routine
    n_sources = len(sources)
    batch_size = n_workers * 1
    refed_paths = set()
    for i in tqdm(range(1 + n_sources // batch_size), leave=False):
        start_loc = max([i * batch_size, 0])
        end_loc = min([(i + 1) * batch_size, n_sources])
        this_sources = sources[start_loc:end_loc]
        try:
            results = executor.map(partial(_load, schema=schema, guard_func=Feature.load), this_sources)
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
        if getattr(type(feature), "after_create", Feature.after_create) != Feature.after_create:
            feature.after_create(db, key)
            f.flush()
    # voila!
    return cls(filename, mode if mode != 'w' else "r+", keep_open)


def _compute(fdict, proxy, parallelism, n_workers, destination):
    sources = [src for src in proxy.owner.__src__.id[proxy.refs[:].astype(np.bool)]]
    executor = get_executor(n_workers, parallelism)
    n_sources = len(sources)
    batch_size = n_workers * 1
    for i in tqdm(range(1 + n_sources // batch_size), leave=False):
        start_loc = max([i * batch_size, 0])
        end_loc = min([(i + 1) * batch_size, n_sources])
        this_sources = sources[start_loc:end_loc]
        if parallelism in ('mp', 'none'):
            res = executor.map(partial(apply_and_store, fdict=fdict, proxy=proxy), this_sources)
        elif parallelism == 'threads':
            res = executor.map(partial(apply_and_store, fdict=fdict, proxy=proxy), this_sources)
        else:
            raise NotImplementedError
        for src, r in res:
            destination.add(src, r)
    if parallelism == 'mp':
        executor.close()
        executor.join()
    return
