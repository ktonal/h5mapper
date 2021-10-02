import torch
import torch.nn as nn
import h5py
from h5mapper import *

from .utils import *


def test_feature_class():
    f = Array()
    with pytest.raises(RuntimeError):
        f.load(None)

    class NotDB(object):
        f = Array()

    o = NotDB()
    with pytest.raises(RuntimeError):
        getattr(o, "f")

    class DB(TypedFile):
        f = Array()

    db = in_mem(DB)
    # print(vars(db.f))
    db.add("0", {"f": np.random.randn(3, 4, 5)})
    assert isinstance(db.f, Proxy)
    assert getattr(db.f, "feature") is DB.f
    assert "Proxy" in repr(db.f)

    class WithTransforms(Array):
        __t__ = (None,)
        __grp_t__ = (None,)

        def __init__(self):
            self.__t__ = WithTransforms.__t__
            self.__grp_t__ = WithTransforms.__grp_t__

    f = WithTransforms()
    # pop the transform when pickling
    state = f.__getstate__()
    assert "__t__" not in state
    assert "__grp_t__" not in state

    # attach the transforms when loading pickles
    f.__setstate__(state)
    assert "__t__" in f.__dict__
    assert "__grp_t__" in f.__dict__

    # default after_create does nothing
    rv = f.after_create(None, None)
    assert rv is None


def test_non_loading_feature(tmp_db):
    class DB(TypedFile):
        x = FeatWithAttrs()

    sources = tuple(map(str, range(20)))
    with pytest.raises(NotImplementedError):
        db = DB.create(tmp_db / "test1.h5", sources,
                       parallelism="mp")
    # for exiting the context
    assert True


def test_none_feature(tmp_db):
    class DB(TypedFile):
        x = NoneFeat()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources,
                   parallelism="mp")

    assert hasattr(db, 'x')
    assert db.x.is_group
    db.info()
    assert not any(isinstance(v, Proxy) for v in db.x.__dict__.values())


def test_bad_feat(tmp_db):
    class DB(TypedFile):
        x = BadFeat()

    sources = tuple(map(str, range(20)))
    with pytest.raises(TypeError):
        db = DB.create(tmp_db / "test1.h5", sources,
                       parallelism="mp")
    import os
    # the file should be gone
    assert not os.path.exists(tmp_db / "test1.h5")


def test_group(tmp_db):
    class DB(TypedFile):
        g = Group(
            x=RandnArray(),
            y=RandnArray(),
            after=AfterFeature()
        )

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources,
                   parallelism="mp")
    assert isinstance(db, DB)
    got = db.get("0")
    assert all(k in got["g"] for k in ["x", "y", "after"]), list(got.keys())

    # can load
    assert isinstance(DB.g.load("any"), dict)


def test_state_dict(tmp_path):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc = nn.Linear(32, 16)
            self.cv = nn.Conv1d(4, 8, 2)

    net = Net()
    torch.save(net.state_dict(), tmp_path / "0.ckpt")
    torch.save(net.state_dict(), tmp_path / "1.ckpt")
    torch.save(net.state_dict(), tmp_path / "2.ckpt")

    class DB(TypedFile):
        # pass the state_dict for initialization
        sd = TensorDict(net.state_dict())

    source = [str(tmp_path / (str(i) + ".ckpt")) for i in range(3)]

    loaded = DB.sd.load(source[0])
    assert all(isinstance(x, np.ndarray) for x in loaded.values())

    db = DB.create(tmp_path / "sd.h5", source)
    assert isinstance(db, DB)
    got = db.get(list(db.index.keys())[0])["sd"]
    assert "fc.weight" in got and "cv.weight" in got
    net.load_state_dict(got)
    assert isinstance(net, Net)

    # all weights are stored with compression
    grp = db.handle(None)[db.sd.name]

    def is_compressed(name, obj):
        if isinstance(obj, h5py.Dataset) and "__arr__" in name:
            assert obj.compression == "lzf"
            assert obj.chunks

    grp.visititems(is_compressed)

    db.info()


def test_vshape(tmp_path):
    class DB(TypedFile):
        x = VShape(RandnArray())

    sources = tuple(map(str, range(8)))
    db = DB.create(tmp_path / "test1.h5", sources,
                   parallelism="mp")

    loaded = DB.x.load("none")
    assert isinstance(loaded, dict)
    assert "arr" in loaded and "shape_" in loaded

    got = db.get("0")
    assert got["x"].shape == RandnArray().load(None).shape


def test_array_transform(tmp_path):
    class DB(TypedFile):
        x = RandnArrayWithT()

    sources = tuple(map(str, range(8)))
    db = DB.create(tmp_path / "test1.h5", sources,
                   parallelism="mp")

    got = db.get("0")
    assert got["x"].shape == RandnArray().load(None).flat[:].shape


def test_vocabulary(tmp_path):
    class DB(TypedFile):
        x = IntVector()
        v = Vocabulary(derived_from="x")

    sources = tuple(map(lambda s: str(s) + '___', range(8)))

    before = DB.v.V.copy()
    DB.v.load(DB.x.load("none"))
    assert before != DB.v.V

    db = DB.create(tmp_path / "test1.h5", sources,
                   parallelism="mp")

    # the dict as it has been loaded
    assert isinstance(db.v.V, dict)
    # the arrays for keys and values
    assert isinstance(db.v.get("data"), dict)
    assert isinstance(db.v.i, Proxy)
    assert isinstance(db.v.x, Proxy)
    # the dict property
    assert isinstance(db.v.dict, dict)
    assert all(i in db.v.dict for i in db.v.i[:])


def test_dir_labels(tmp_path):
    class DB(TypedFile):
        x = IntVector()
        v = DirLabels()

    class LoadDB(TypedFile):
        x = IntVector()
        v = DirLabels()

    sources = tuple(map(lambda s: str(s) + '___/', range(8)))
    DB.create(tmp_path / "test1.h5", sources,
              parallelism="mp")
    db = LoadDB(tmp_path / "test1.h5")
    assert all(k + "/" in sources for k in db.v.d2i.keys()), db.v.d2i


def test_files_labels(tmp_path):
    class DB(TypedFile):
        x = IntVector()
        v = FilesLabels(derived_from='x')

    class LoadDB(TypedFile):
        x = IntVector()
        v = FilesLabels(derived_from='x')

    sources = tuple(map(lambda s: str(s) + '___', range(8)))
    DB.create(tmp_path / "test1.h5", sources,
              parallelism="mp")
    db = LoadDB(tmp_path / "test1.h5")
    assert db.v.shape[0] == db.x.shape[0], (db.v.shape, db.x.shape)
    assert all(k in sources for k in db.v.f2i.keys()), db.v.f2i


def test_rand_string(tmp_path):
    class DB(TypedFile):
        s = RandString()

    sources = tuple(map(lambda s: str(s) + '___', range(8)))
    db = DB.create(tmp_path / "test1.h5", sources,
                   parallelism="mp")

    assert db.s.asstr
    assert all(isinstance(s, str) for s in db.s[:])
