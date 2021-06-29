# H5Mapper

The H5Mapper is a pythonic ORM tool for reading and writing HDF5 data.
It is built on top of `h5py` and simply maps the hierarchy of a HDF5 file to a python object and its attributes or the other way around : 
creates a `.h5` file from a list of sources (e.g. paths, urls) and a python class that determines the 'schema' of the resulting HDF5 file.

The H5Mapper is intended for Machine Learning applications where, more than often, one needs to
    
- load different kinds of data (e.g. images and captions) from raw sources/files efficiently

- have a clear management of loading parameters (e.g. sample rate) for downstream tasks

- switch between indexing schemes (e.g. sources are primary keys vs. integer based indexing in the concatenation of all sources)

- read and write data for different purposes (datasets, model checkpoints, losses...) without having to implement transactions with a file or getting lost in the file system layout of experiments...

The H5Mapper tries to simplify all that by 
 
***expressing the full lifecycle of your ML data (from creation to consumption) in a single, simple and efficient class***


## Usage 

One needs only 2 classes to do so :

1. a `Array` must implement `load(source)` which can return a `numpy.ndarray`, a `pandas.DataFrame` or a `dict` thereof with strings as keys.

```python
import h5mapper as h5m


class MyFeature(h5m.Array):
    
    def __init__(self, my_extraction_param=0):
        self.my_extraction_param = my_extraction_param

    @property
    def attrs(self):
        # those are then written to the file
        return {"p": self.my_extraction_param}

    def load(self, source):
        # your method to get an np.ndarray, a pd.DataFrame or any dict thereof
        # from a path, an url, whatever sources you have...   
        return data

    def plot(self, data):
        # custom plotting method for this kind of data
        # ...
```

2. attach the `Array` to a class inheriting from a `h5m.FileType` and that's it!

```python

class MyDB(h5m.FileType):
    
    x = MyFeature(my_extraction_param=42)


if __name__ == "__main__":
    # run a parallel extraction job on your sources
    db = MyDB.create("dataset.h5", my_list_of_sources)

    # read your data through __getitem__ 
    batch = db.x[4:8]

    # access your method 
    db.x.plot(batch)

    # modify the file through __setitem__
    db.x[4:8] = batch ** 2 
```

Primarly designed with `pytorch` users in mind, `h5m` plays very nicely with the `Dataset` class :

```python
class MyDS(h5m.FileType, torch.utils.data.Dataset):
    
    x = MyInputFeature(42)
    labels = FilesLabels()
    
    def __getitem__(self, item):
        return self.x[item], self.labels[item]
  
    def __len__(self):
        return len(self.x)

# create (`keep_open=True` makes sure the file stays open *within* each process later spawn by the loader)
ds = MyDS.create("train.h5", sources, keep_open=True)

# load batches as fast as it gets
dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=8, pin_memory=True)
```

### Coming soon

The H5Mapper just started, but soon, it will feature :

- full CRUD (for now, one can only read, add to a file or modify it in-place)
- standard feature classes for images, audios, state_dict...
- docs with more examples 
- more tests

If you'd like to help, just drop us an email : ktonalberlin@gmail.com


### License

`H5Mapper` is distributed under the terms of the MIT License. 