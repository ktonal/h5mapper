import pytest
import torch
import torch.nn as nn

from h5m import *

from .utils import *


def test_feature_class():
    f = Feature()
    with pytest.raises(RuntimeError):
        f.load(None)

    class NotDB(object):
        f = Feature()

    o = NotDB()
    with pytest.raises(RuntimeError):
        getattr(o, "f")

    class DB(Database):
        f = Feature()

    db = in_mem(DB)
    db.add("0", {"f": np.random.randn(3, 4, 5)})
    assert isinstance(db.f, Proxy)
    assert getattr(db.f, "feature") is DB.f
    assert "Proxy" in repr(db.f)

    class WithTransforms(Feature):
        __t__ = (None, )
        __grp_t__ = (None, )

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
    class DB(Database):
        x = FeatWithAttrs()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources,
                   parallelism="mp")

    assert hasattr(db, 'x')
    assert not any(isinstance(v, Proxy) for v in db.x.__dict__.values())
    assert db.load("34") == {}


def test_none_feature(tmp_db):
    class DB(Database):
        x = NoneFeat()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources,
                   parallelism="mp")

    assert hasattr(db, 'x')
    assert db.x.is_group
    assert not any(isinstance(v, Proxy) for v in db.x.__dict__.values())


def test_bad_feat(tmp_db):
    class DB(Database):
        x = BadFeat()

    sources = tuple(map(str, range(20)))
    with pytest.raises(TypeError):
        db = DB.create(tmp_db / "test1.h5", sources,
                       parallelism="mp")
    import os
    # the file should be gone
    assert not os.path.exists(tmp_db / "test1.h5")


def test_group(tmp_db):
    class DB(Database):
        g = Group(
            x=Array(),
            y=DF(),
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


# TODO
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

    class DB(Database):
        # pass the state_dict for initialization
        sd = StateDict(net.state_dict())

    source = [str(tmp_path / (str(i) + ".ckpt")) for i in range(3)]

    loaded = DB.sd.load(source[0])
    assert all(isinstance(x, np.ndarray) for x in loaded.values())

    db = DB.create(tmp_path / "sd.h5", source)
    assert isinstance(db, DB)
    got = db.get(db.sd.ids[0])["sd"]
    assert "fc.weight" in got and "cv.weight" in got
    net.load_state_dict(got)
    assert isinstance(net, Net)

    # all weights are stored with compression
    grp = db.handler("h5py", None)[db.sd.name]

    def is_compressed(name, obj):
        if isinstance(obj, h5py.Dataset) and "__arr__" in name:
            assert obj.compression == "lzf"
            assert obj.chunks

    grp.visititems(is_compressed)

    db.info()


def test_vshape(tmp_path):
    class DB(Database):
        x = VShape(Array())

    sources = tuple(map(str, range(8)))
    db = DB.create(tmp_path / "test1.h5", sources,
                   parallelism="mp")

    loaded = DB.x.load("none")
    assert isinstance(loaded, dict)
    assert "arr" in loaded and "_shape" in loaded

    got = db.get("0")
    assert got["x"].shape == Array().load(None).shape


def test_array_transform(tmp_path):
    class DB(Database):
        x = ArrayWithT()

    sources = tuple(map(str, range(8)))
    db = DB.create(tmp_path / "test1.h5", sources,
                   parallelism="mp")

    got = db.get("0")
    assert got["x"].shape == Array().load(None).flat[:].shape


def test_vocabulary(tmp_path):
    class DB(Database):
        x = IntVector()
        v = Vocabulary(derived_from="x")

    sources = tuple(map(str, range(8)))

    before = DB.v.V.copy()
    DB.v.load(DB.x.load("none"))
    assert before != DB.v.V

    db = DB.create(tmp_path / "test1.h5", sources,
                   parallelism="mp")

    # the dict as is has been discovered
    assert isinstance(db.v.V, dict)
    # the arrays for keys and values
    assert isinstance(db.v.get("xi"), dict)
    assert isinstance(db.v.i, Proxy)
    assert isinstance(db.v.x, Proxy)
