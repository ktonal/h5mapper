import pytest
import numpy as np
import pandas as pd

from h5mapper.h5m import *


class Array(Feature):
    __ext__ = {"any"}

    def load(self, source):
        return np.random.randn(4, 12)


class DF(Feature):
    __ext__ = {"any"}

    def load(self, source):
        return pd.DataFrame(np.random.randn(4, 12))


class Dict(Feature):
    __ext__ = {"any"}

    def load(self, source):
        return dict(x=np.random.randn(4, 12),
                    y=pd.DataFrame(np.random.randn(4, 12))
                    )


class DictofDict(Feature):
    __ext__ = {"any"}

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
    check_array_feature(db.x)

    db = DB.create(tmp_db / "test2.h5", sources,
                   parallelism="future")

    assert isinstance(db, DB)
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
    check_df_feature(db.x)

    db = DB.create(tmp_db / "test2.h5", sources,
                   parallelism="future")

    assert isinstance(db, DB)
    check_df_feature(db.x)


def check_dict_feature(feat):
    check_array_feature(feat.x)
    check_df_feature(feat.y)
    # get a dict of data for source "0"
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
    check_dict_feature(db.D)

    db = DB.create(tmp_db / "test1.h5", sources, parallelism='future')

    assert isinstance(db, DB)
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
    check_dictofdict_feature(db.D)

    db = DB.create(tmp_db / "test1.h5", sources, parallelism='future')

    assert isinstance(db, DB)
    check_dictofdict_feature(db.D)
