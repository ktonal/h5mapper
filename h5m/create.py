import h5py
from multiprocessing import cpu_count, Pool
from concurrent.futures import ThreadPoolExecutor
import os
from functools import partial
from tqdm import tqdm

from .features import Array
from .crud import _add, _load, H5_NONE

__all__ = [
    "_create"
]


def _create(cls,
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
        schema = {attr: val for attr, val in cls.__dict__.items() if isinstance(val, Array)}
    if not schema:
        raise ValueError("schema cannot be empty. Either provide one to create()"
                         " or attach Array attributes to this class.")
    # create two separate files for arrays and dataframes
    f = h5py.File(h5_file, mode, **h5_kwargs)

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
    for i in tqdm(range(1 + n_sources // batch_size)):
        start_loc = max([i * batch_size, 0])
        end_loc = min([(i + 1) * batch_size, n_sources])
        this_sources = sources[start_loc:end_loc]
        try:
            # we use FileType.load instead of cls.load because cls might be in a <locals>
            # which Pool can not pickle...
            results = executor.map(partial(_load, schema=schema, guard_func=Array.load), this_sources)
        except Exception as e:
            f.flush()
            f.close()
            os.remove(h5_file)
            if parallelism == 'mp':
                executor.terminate()
            raise e
        # write results
        for n, res in enumerate(results):
            for key, data in res.items():
                if data is None:
                    continue
                _add.source(groups[key], this_sources[n], data, ds_kwargs[key])
                # f.flush()
    if parallelism == 'mp':
        executor.close()
        executor.join()
    f.flush()

    # run after_create
    db = cls(h5_file, mode="r+", keep_open=False)
    for key, feature in schema.items():
        if getattr(type(feature), "after_create", Array.after_create) != Array.after_create:
            feature.after_create(db, key)
            f.flush()
    # voila!
    return cls(h5_file, mode if mode != 'w' else "r+", keep_open)
