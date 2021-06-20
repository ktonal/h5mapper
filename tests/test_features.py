import pytest
from h5m import *
from h5m.features import Group

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


# TODO
def test_state_dict():
    pass


def test_vshape():
    pass


def test_vocabulary():
    pass

