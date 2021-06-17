import pytest
import numpy as np
import pandas as pd

from h5m import *


def test_feature_class():
    f = Feature()
    with pytest.raises(RuntimeError):
        f.load(None)
    with pytest.raises(RuntimeError):
        f.after_create(None, None)

    class NotDB(object):
        f = Feature()

    o = NotDB()
    with pytest.raises(RuntimeError):
        getattr(o, "f")


class FeatWithAttrs(Feature):
    __re__ = r".*"

    param1 = 18
    param2 = "hey"
    param3 = None

    @property
    def attrs(self):
        return dict(p1=self.param1, p2=self.param2, p3=self.param3)


class NoneFeat(Feature):

    def load(self, source):
        return None


class BadFeat(Feature):

    def load(self, source):
        return "this should raise an Exception"


# Todo : parametrize test with a couple of dtypes, __ds_kwargs__
class Array(FeatWithAttrs):

    def load(self, source):
        return np.random.randn(4, 12)


class DF(FeatWithAttrs):

    def load(self, source):
        return pd.DataFrame(np.random.randn(4, 12))


class Dict(FeatWithAttrs):

    def load(self, source):
        return dict(x=np.random.randn(4, 12),
                    y=pd.DataFrame(np.random.randn(4, 12))
                    )


class DictofDict(FeatWithAttrs):

    def load(self, source):
        return dict(
            p=dict(x=np.random.randn(4, 12),
                   y=pd.DataFrame(np.random.randn(4, 12))
                   ),
            q=dict(x=np.random.randn(4, 12),
                   y=pd.DataFrame(np.random.randn(4, 12))
                   )
        )


@pytest.fixture
def tmp_db(tmp_path):
    root = (tmp_path / "dbs")
    root.mkdir()
    return root


def check_db(db):
    assert isinstance(db, Database)
    copy = Database(db.h5_file)
    copy.keep_open = True
    # check that there is only one handler that stays open
    h1 = copy.handler('h5py', 'r')
    h2 = copy.handler('h5py', 'r')
    s1 = copy.handler('pd', 'r')
    s2 = copy.handler('pd', 'r')
    assert h1 is h2 and s1 is s2
    copy.close()
    # handle should now be closed
    assert not h1 and not s1.is_open

    copy.keep_open = False
    # check that there is multiple handler
    h1 = copy.handler('h5py', 'r')
    h2 = copy.handler('h5py', 'r')
    s1 = copy.handler('pd', 'r')
    s2 = copy.handler('pd', 'r')
    assert h1 is not h2 and s1 is not s2
    copy.close()
    # should not affect handlers
    assert h1 and h2 and s1.is_open and s2.is_open
    h1.close()
    h2.close()
    s1.close()
    s2.close()

    # can load with no schema
    rv = db.load("42")
    assert isinstance(rv, dict)


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


def check_feature_attrs(feat):
    assert feat.attrs == FeatWithAttrs().attrs


def check_array_feature(feat):
    assert isinstance(feat[:], np.ndarray)
    # Proxies that should be hosted by the feature
    assert hasattr(feat, 'src') and isinstance(feat.src, Proxy)
    assert hasattr(feat, 'refs') and isinstance(feat.refs, Proxy)
    assert hasattr(feat, 'ids') and isinstance(feat.ids, Proxy)
    # since ids were strings, they should be wrapped in .asstr()
    assert feat.ids.asstr and isinstance(feat.ids[0], str)
    # get the array for the (integer-based) source 0
    assert isinstance(feat[feat.refs[0]], np.ndarray)
    assert feat[feat.refs[0]].shape == Array().load(None).shape

    # check that keeping the db open issues only one handler
    feat.owner.keep_open = True
    hf = feat.handler()
    assert hf is feat.owner.handler(feat.kind, 'r')
    # should now be opened
    assert isinstance(feat[:], np.ndarray)
    hf.close()
    assert not feat.owner._f
    feat.owner.keep_open = False



def test_create_arrays(tmp_db):
    class DB(Database):
        x = Array()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources,
                   parallelism="mp")

    check_db(db)
    check_feature_attrs(db.x)
    check_array_feature(db.x)

    db = DB.create(tmp_db / "test2.h5", sources,
                   parallelism="future")

    check_db(db)
    check_feature_attrs(db.x)
    check_array_feature(db.x)


def check_df_feature(feat):
    assert isinstance(feat[:], pd.DataFrame)
    # the 'source' Index has been added
    assert isinstance(feat[:].index.get_level_values("source"), pd.Index)
    # the df for (label-based) source "0" has correct shape
    assert feat[:].loc["0"].shape == DF().load(None).shape
    # no Proxy is attached for the children added by the HDFStore
    assert not hasattr(feat, "_i_table")
    assert not hasattr(feat, "table")


def test_create_dataframes(tmp_db):
    class DB(Database):
        x = DF()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources,
                   parallelism="mp")

    check_db(db)
    check_feature_attrs(db.x)
    check_df_feature(db.x)

    db = DB.create(tmp_db / "test2.h5", sources,
                   parallelism="future")

    check_db(db)
    check_feature_attrs(db.x)
    check_df_feature(db.x)


def check_dict_feature(feat):
    check_array_feature(feat.x)
    check_df_feature(feat.y)
    # Proxies that should be hosted by the feature
    assert hasattr(feat, 'src') and isinstance(feat.src, Proxy)
    assert hasattr(feat.src, 'refs') and isinstance(feat.src.refs, Proxy)
    assert hasattr(feat.src, 'ids') and isinstance(feat.src.ids, Proxy)
    assert hasattr(feat.src, 'keys') and isinstance(feat.src.keys, Proxy)
    assert hasattr(feat, 'keys') and isinstance(feat.keys, Proxy)
    assert hasattr(feat, 'ids') and isinstance(feat.ids, Proxy)
    assert hasattr(feat, 'refs') and isinstance(feat.refs, Proxy)
    # get the dict of data for source "0"
    src_data = feat["0"]
    assert isinstance(src_data, dict)
    assert "x" in src_data and "y" in src_data
    assert src_data["x"].shape == Array().load(None).shape
    assert src_data["y"].shape == DF().load(None).shape

    # exception when indexing groups with not str
    with pytest.raises(TypeError):
        ouch = feat[0]

def test_create_dict(tmp_db):
    class DB(Database):
        D = Dict()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources, parallelism='mp')

    check_db(db)
    check_feature_attrs(db.D)
    check_dict_feature(db.D)

    db = DB.create(tmp_db / "test1.h5", sources, parallelism='future')

    check_db(db)
    check_feature_attrs(db.D)
    check_dict_feature(db.D)


def check_dictofdict_feature(feat):
    check_dict_feature(feat.p)
    check_dict_feature(feat.q)
    # get a dict of dict of data for source "0"
    src_data = feat["0"]
    assert isinstance(src_data, dict)
    assert "p" in src_data and "q" in src_data


def test_create_dictofdict(tmp_db):
    class DB(Database):
        D = DictofDict()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources, parallelism='mp')

    check_db(db)
    check_feature_attrs(db.D)
    check_dictofdict_feature(db.D)

    db = DB.create(tmp_db / "test1.h5", sources, parallelism='future')

    check_db(db)
    check_feature_attrs(db.D)
    check_dictofdict_feature(db.D)


#### DERIVED ARRAY

class DerivedArray(FeatWithAttrs):

    def __init__(self, derived_from="x"):
        self.derived_from = derived_from

    def load(self, source):
        return source + 1


def test_derived_array(tmp_db):
    class DB(Database):
        x = Array()
        y = DerivedArray(derived_from="x")

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources, parallelism='mp')

    check_db(db)
    check_feature_attrs(db.x)
    check_array_feature(db.x)
    check_array_feature(db.y)
    assert np.all(db.x.ids[()] == db.y.ids[()])
    # somehow they are not strictly equal...
    assert np.allclose(db.x[:], (db.y[:] - 1))


class AfterFeature(FeatWithAttrs):

    def load(self, source):
        return Array().load(source)

    def after_create(self, db, feature_key):
        print(self, feature_key)
        feat = db.get_feat(feature_key)
        # modify the feature in-place (file should be open for write)
        feat[:] = np.zeros_like(feat[:])
        return None


def check_after_feature(feat):
    assert np.all(feat[:] == 0)


# TODO : What does one want after load?
#  Build vocabularies?...
#  Change dtype, e.g. [0, 255] -> [0., 1.]?...
#  Normalize with mean and std?...
class AfterArray(FeatWithAttrs):
    def load(self, source):
        return None

    def after_create(self, db, feature_key):
        return Array().load(None)


class AfterDF(FeatWithAttrs):
    def load(self, source):
        return None

    def after_create(self, db, feature_key):
        return DF().load(None)


class AfterDict(FeatWithAttrs):
    def load(self, source):
        return None

    def after_create(self, db, feature_key):
        return {"x": Array().load(None), "y": DF().load(None)}


def test_after_create(tmp_db):

    class DB(Database):
        x = AfterFeature()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources, parallelism='mp')

    check_db(db)
    check_after_feature(db.x)


def test_after_array(tmp_db):

    class DB(Database):
        x = AfterArray()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources, parallelism='mp')

    check_db(db)


def test_after_df(tmp_db):

    class DB(Database):
        x = AfterDF()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources, parallelism='mp')

    check_db(db)


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
    check_dict_feature(db.g)
    check_after_feature(db.g.after)

    db = DB.create(tmp_db / "test2.h5", sources,
                   parallelism="future")

    check_db(db)
    check_feature_attrs(db.g)
    # currently, attrs are not added recursively...
    with pytest.raises(AssertionError):
        check_feature_attrs(db.g.x)
    check_dict_feature(db.g)
    check_after_feature(db.g.after)
