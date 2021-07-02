# h5mapper

``h5mapper`` is a pythonic ORM-like tool for reading and writing HDF5 data.

It is built on top of `h5py` and lets you define types of **.h5 files as python classes** which you can then easily 
**create from raw sources** (e.g. files, urls...), **serve** (use as ``Dataset`` for a ``Dataloader``), 
or dynamically populate (logs, checkpoints of an experiment).

## Content
- [Installation](#Installation)
- [Quickstart](#Quickstart)
    - [TypedFile](#TypedFile)
    - [Feature](#Feature)
- [Examples](#Examples)
- [Development](#Development)
- [License](#License)
 
## Installation

### ``pip``

``h5mapper`` is on pypi, to install it, one only needs to 

```bash
pip install h5mapper
```

### developer install

for playing around with the internals of the package, a good solution is to first

```bash
git clone https://github.com/ktonal/h5mapper.git
```
and then 

```bash
pip install -e h5mapper/
```
which installs the repo in editable mode.

## Quickstart

### TypedFile

``h5m`` assumes that you want to store collections of contiguous arrays in single datasets and that you want several such concatenated datasets in a file.

Thus, ``TypedFile`` allows you to create and read files that maintain a 2-d reference system, where contiguous arrays are stored within features and indexed by their source's id.

Such a file might then look like 
```bash
<Experiment "experiment.h5">
----------------------------------------------------> sources' ids axis
|                   "planes/01.jpeg"  |     "train"
|                                     |
|   data/                             |
|        images/        (32, 32)      |       None
|        labels/        (1, )         |       None
|   logs/                             |
|        loss/           None         |       (10000,)
|        ...
V
features axis
``` 
where the entries correspond to the shapes of arrays or their absence (`None`).

> Note that this is a different approach than storing each file or image in a separate dataset. 
> In this case, there would be an `h5py.Dataset` located at `data/images/planes/01.jpeg` although in our
> example, the only dataset is at `data/images/` and one of its regions is indexed by the id `"planes/01.jpeg"` 

For interacting with files that follow this particular structure, simply define a class

```python
import h5mapper as h5m

class Experiment(h5m.TypedFile):

    data = h5m.Group(
            images=h5m.Image(),
            labels=h5m.DirLabels()
            )
    logs = h5m.Group(
            loss=h5m.Array()
            )
```
#### ``create``, ``add``

now, create an instance, load data from files through parallel jobs and add data on the fly :

```python
# create instance from raw sources
exp = Experiment.create("experiment.h5",
        # those are then used as ids :
        sources=["planes/01.jpeg", "planes/02.jpeg"],
        n_workers=8)
...
# add id <-> data on the fly :
exp.logs.add("train", dict(loss=losses_array))
``` 

#### ``get``, ``refs`` and ``__getitem__`` 

There are 3 main options to read data from a ``TypedFile`` or one of its ``Proxy``

1/ By their id

```python
>> exp.logs.get("train")
Out: {"loss": np.array([...])}
# which, in this case, is equivalent to 
>> exp.logs["train"]
Out: {"loss": np.array([...])}
# because `exp.logs` is a Group and Groups only support id-based indexing
```

2/ By the index of their ids through their ``refs`` attribute :

```python
>> exp.data.images[exp.data.images.refs[0]].shape
Out: (32, 32)
```
Which works because `exp.data.images` is a `Dataset` and only `Datasets` have `refs`

3/ with any ``item`` supported by the ``h5py.Dataset``
```python
>> exp.data.labels[:32]
Out: np.array([0, 0, ....])
```
Which also only works for `Datasets`.

> Note that, in this last case, you are indexing into the **concatenation of all sub-arrays along their first axis**.

> The same interface is also implemented for ``set(source, data)`` and ``__setitem__``

### Feature

``h5m`` exposes a class that helps you configure the behaviour of your ``TypedFile`` classes and the properties of the .h5 they create.

the ``Feature`` class helps you define :
- how sources' ids are loaded into arrays (``feature.load(source)``)
- which types of files are supported
- how the data is stored by ``h5py`` (compression, chunks)
- which extraction parameters need to be stored with the data (e.g. sample rate of audio files)
- custom-methods relevant to this kind of data

Once you defined a `Feature` class, attach it to the class dict of a ``TypedFile``, that's it!

For example :

```python
import h5mapper as h5m


class MyFeature(h5m.Feature):

    # only sources matching this pattern will be passed to load(...)
    __re__ = r".special$"

    # args for the h5py.Dataset
    __ds_kwargs__ = dict(compression='lzf', chunks=(1, 350))
    
    def __init__(self, my_extraction_param=0):
        self.my_extraction_param = my_extraction_param

    @property
    def attrs(self):
        # those are then written in the h5py.Group.attrs
        return {"p": self.my_extraction_param}

    def load(self, source):
        """your method to get an np.ndarray or a dict thereof
        from a path, an url, whatever sources you have..."""   
        return data

    def plot(self, data):
        """custom plotting method for this kind of data"""
        # ...

# attach it
class Data(h5m.TypedFile):
    feat = MyFeature(47)

# load sources...
f = Data.create(....)

# read your data through __getitem__ 
batch = f.feat[4:8]

# access your method 
f.feat.plot(batch)

# modify the file through __setitem__
f.feat[4:8] = batch ** 2 
```

for more examples, checkout `h5mapper/h5mapper/features.py`.

#### ``serve``

Primarly designed with `pytorch` users in mind, `h5m` plays very nicely with the `Dataset` class :

```python
class MyDS(h5m.TypedFile, torch.utils.data.Dataset):
    
    x = MyInputFeature(42)
    labels = h5m.DirLabels()
    
    def __getitem__(self, item):
        return self.x[item], self.labels[item]
  
    def __len__(self):
        return len(self.x)

ds = MyDS.create("train.h5", sources, keep_open=True)

dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=8, pin_memory=True)
```

`TypedFile` even have a method that takes the Dataloader args and a batch object filled with `BatchItems` and returns 
a Dataloader that will yield such batch objects.

Example :

```python
f = TypedFile("train.h5", keep_open=True)
loader = f.serve(
    # batch object :
    dict(
        x=h5m.Input(key='data/image', getter=h5m.GetId()),
        labels=h5m.Target(key='data/labels', getter=h5m.GetId())
    ),
    # Dataloader kwargs :
    num_workers=8, pin_memory=True, batch_size=32, shuffle=True
)
```  

### Examples

in ``h5mapper/examples`` you'll find for now
- a train script with data, checkpoints and logs in `dataset_and_logs.py`
- two click command-lines for making image- and soundbanks
- a script for benchmarking batch-loading times of different options

### Development

`h5mapper` is just getting started and you're welcome to contribute!

You'll find some tests you can run from the root of the repo with a simple
```bash
pytest
```

If you'd like to get involved, just drop us an email : ktonalberlin@gmail.com


### License

`h5mapper` is distributed under the terms of the MIT License. 