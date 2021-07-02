import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import os

import h5mapper as h5m

# last dim of data
D = 256


class RandnArray(h5m.Array):

    def __init__(self, n, **ds_kwargs):
        self.n = n
        self.__ds_kwargs__.update(ds_kwargs)

    def load(self, source):
        return np.random.randn(self.n, D).astype(np.float32)


def get_loader(compression=None, slice_length=16, keep_open=True):

    class Data(h5m.TypedFile):
        x = RandnArray(slice_length, compression=compression, chunks=(slice_length, D))
        y = RandnArray(slice_length, compression=compression, chunks=(slice_length, D))
        z = RandnArray(slice_length, compression=compression, chunks=(slice_length, D))

    ds = Data.create("bench-serve.h5", list(map(str, range(16*500))), keep_open=keep_open)

    return ds, ds.serve(
        # batch object
        dict(x=h5m.Input(key="x", getter=h5m.GetId()),
             y=h5m.Input(key="y", getter=h5m.GetId()),
             z=h5m.Input(key="z", getter=h5m.GetId())
             ),
        # loaders kwargs
        shuffle=True,
        num_workers=16,
        batch_size=16,
        pin_memory=True,
        prefetch_factor=2
    )


if __name__ == '__main__':
    avg_time = {}
    for comp in [None, 'lzf', 'gzip']:
        for ko in [True, False]:
            avg_time[(comp, ko)] = {}
            for length in [16, 64, 128, 256]:
                file, loader = get_loader(compression=comp, slice_length=length, keep_open=ko)
                # get the 1st one out
                _ = next(iter(loader))
                times = []
                before = time()
                for _ in tqdm(loader, leave=False):
                    now = time()
                    times += [now - before]
                    before = now
                avg_time[(comp, ko)][length] = sum(times) / len(times)
                file.close()
    fig, ax = plt.subplots()
    for comp, ko in avg_time.keys():
        avg = avg_time[(comp, ko)]
        ax.scatter(list(avg.keys()), list(avg.values()), label=f"compression={comp}, keep_open={ko}",
                   marker="x", linewidths=1.)
    ax.set_xlabel("slice's length")
    ax.set_ylabel("avg load time / batch")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()
    os.remove("bench-serve.h5")
