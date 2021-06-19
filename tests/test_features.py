import pytest
from h5m import Feature, Database, Proxy
from h5m.features import Group
from .test_core import check_db
from .test_create import check_feature_attrs, check_dict_proxy, \
    check_after_feature
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

    check_db(db)
    check_feature_attrs(db.g)
    # currently, attrs are not added recursively...
    with pytest.raises(AssertionError):
        check_feature_attrs(db.g.y)
    check_dict_proxy(db.g)
    check_after_feature(db.g.after)

    db = DB.create(tmp_db / "test2.h5", sources,
                   parallelism="future")

    check_db(db)
    check_feature_attrs(db.g)
    # currently, attrs are not added recursively...
    with pytest.raises(AssertionError):
        check_feature_attrs(db.g.x)
    check_dict_proxy(db.g)
    check_after_feature(db.g.after)