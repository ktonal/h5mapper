import pytest
import numpy as np
import pandas as pd

from h5m import *


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


class AfterFeature(FeatWithAttrs):

    def load(self, source):
        return Array().load(source)

    def after_create(self, db, feature_key):
        feat = getattr(db, feature_key)
        # modify the feature in-place (file should be open for write)
        feat[:] = np.zeros_like(feat[:])
        return None


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


@pytest.fixture
def tmp_db(tmp_path):
    root = (tmp_path / "dbs")
    root.mkdir()
    return root


def test_none_feature(tmp_db):
    class DB(Database):
        x = NoneFeat()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources,
                   parallelism="mp")

    assert hasattr(db, 'x')
    assert db.x.is_group
    assert not any(isinstance(v, Proxy) for v in db.x.__dict__.values())


def check_feature_attrs(feat):
    assert feat.attrs == FeatWithAttrs().attrs


def check_array_feature(feat):
    assert isinstance(feat[:], np.ndarray)
    assert hasattr(feat, 'src')
    assert hasattr(feat, 'refs')
    # get the array for the (integer-based) source 0
    assert isinstance(feat[feat.refs[0]], np.ndarray)
    assert feat[feat.refs[0]].shape == Array().load(None).shape


def test_create_arrays(tmp_db):
    class DB(Database):
        x = Array()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources,
                   parallelism="mp")

    assert isinstance(db, DB)
    check_feature_attrs(db.x)
    check_array_feature(db.x)

    db = DB.create(tmp_db / "test2.h5", sources,
                   parallelism="future")

    assert isinstance(db, DB)
    check_feature_attrs(db.x)
    check_array_feature(db.x)


def check_df_feature(feat):
    assert isinstance(feat[:], pd.DataFrame)
    # the 'source' Index has been added
    assert isinstance(feat[:].index.get_level_values("source"), pd.Index)
    # the df for (label-based) source "0" has correct shape
    assert feat[:].loc["0"].shape == DF().load(None).shape


def test_create_dataframes(tmp_db):
    class DB(Database):
        x = DF()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources,
                   parallelism="mp")

    assert isinstance(db, DB)
    check_feature_attrs(db.x)
    check_df_feature(db.x)

    db = DB.create(tmp_db / "test2.h5", sources,
                   parallelism="future")

    assert isinstance(db, DB)
    check_feature_attrs(db.x)
    check_df_feature(db.x)


def check_dict_feature(feat):
    check_array_feature(feat.x)
    check_df_feature(feat.y)
    # get the dict of data for source "0"
    src_data = feat["0"]
    assert isinstance(src_data, dict)
    assert "x" in src_data and "y" in src_data
    assert src_data["x"].shape == Array().load(None).shape
    assert src_data["y"].shape == DF().load(None).shape


def test_create_dict(tmp_db):
    class DB(Database):
        D = Dict()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources, parallelism='mp')

    assert isinstance(db, DB)
    check_feature_attrs(db.D)
    check_dict_feature(db.D)

    db = DB.create(tmp_db / "test1.h5", sources, parallelism='future')

    assert isinstance(db, DB)
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

    assert isinstance(db, DB)
    check_feature_attrs(db.D)
    check_dictofdict_feature(db.D)

    db = DB.create(tmp_db / "test1.h5", sources, parallelism='future')

    assert isinstance(db, DB)
    check_feature_attrs(db.D)
    check_dictofdict_feature(db.D)


def test_after_create(tmp_db):

    class DB(Database):
        x = AfterFeature()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources, parallelism='mp')

    assert isinstance(db, DB)
    assert np.all(db.x[:] == 0)


def test_after_array(tmp_db):

    class DB(Database):
        x = AfterArray()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources, parallelism='mp')

    assert isinstance(db, DB)


def test_after_df(tmp_db):

    class DB(Database):
        x = AfterDF()

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources, parallelism='mp')

    assert isinstance(db, DB)
