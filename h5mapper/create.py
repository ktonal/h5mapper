import h5py
import numpy as np
from multiprocess import cpu_count, Pool
from concurrent.futures import ThreadPoolExecutor
import os
from functools import partial
from IPython import get_ipython

from .features import Feature
from .crud import _add, _load, SRC_KEY, apply_and_store
from .utils import flatten_dict

shell = get_ipython().__class__.__name__
if shell in ('ZMQInteractiveShell', "Shell"):
    # local and colab notebooks
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

__all__ = [
    "_create",
    '_compute',
    "for_each",
    "tqdm"
]


class SerialExecutor:
    def imap(self, func, iterable, **kwargs):
        return map(func, iterable)
    map = imap
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
        # dirty hack...
        executor.imap = executor.map
    elif parallelism == 'none':
        executor = SerialExecutor()
    else:
        raise ValueError(f"parallelism must be one of ['mp', 'threads', 'none']. Got '{parallelism}'")
    return executor


def for_each(iterable, f, parallelism="mp", n_workers=cpu_count(), chunksize=None, **tqdm_kwargs):
    executor = get_executor(n_workers, parallelism)
    if chunksize is None:
        chunksize = max(len(iterable)//n_workers, 1)
    try:
        for result in tqdm(executor.imap(f, iterable, chunksize=chunksize), **tqdm_kwargs):
            yield result
    except Exception as e:
        if parallelism == 'mp':
            executor.terminate()
        raise e
    if parallelism == 'mp':
        executor.close()
        executor.join()


def _create(cls,
            filename,
            sources,
            mode="w",
            schema={},
            n_workers=cpu_count(),
            parallelism='mp',
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
    # create groups from schema
    for key in schema:
        if key not in f:
            f.create_group(key)
    f.flush()
    # initialize ds_kwargs from schema
    ds_kwargs = {key: getattr(feature, "__ds_kwargs__", {}).copy() for key, feature in schema.items()}

    # run loading routine
    refed_paths = set()
    func = partial(_load, schema=schema)
    try:
        for i, result in enumerate(for_each(sources, func, parallelism, n_workers, chunksize=1, leave=True, desc="Extracting Files", unit="file", total=len(sources))):
            if not result:
                continue
            result = flatten_dict(result)
            _add.source(f, sources[i], result, ds_kwargs, refed_paths)
            refed_paths = refed_paths | set(result.keys())
        f.flush()
    except Exception as e:
        f.close()
        if mode == "w":
            os.remove(filename)
        raise e

    # run after_create
    db = cls(filename, mode="r+", keep_open=True)
    for key, feature in schema.items():
        if getattr(type(feature), "after_create", Feature.after_create) != Feature.after_create:
            feature.after_create(db, key)
            f.flush()
    if hasattr(cls, "after_create"):
        db.after_create()
        f.flush()
    db.close()
    # voila!
    f.close()
    return


def _compute(fdict, proxy, parallelism, n_workers, destination):
    sources = [src for src in proxy.owner.__src__.id[proxy.refs[:].astype(bool)]]
    executor = get_executor(n_workers, parallelism)
    n_sources = len(sources)
    batch_size = n_workers * 1
    out = {}
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
            if destination is None:
                out.update({src: r})
            else:
                destination.add(src, r)
    if parallelism == 'mp':
        executor.close()
        executor.join()
    return out if destination is None else destination
