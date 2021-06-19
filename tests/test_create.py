import pytest

from h5m import *
from .utils import *

from .test_core import check_db


# Todo : parametrize test with a couple of dtypes, __ds_kwargs__


def check_feature_attrs(feat):
    assert feat.attrs == FeatWithAttrs().attrs


def check_array_proxy(feat):
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
    check_array_proxy(db.x)

    db = DB.create(tmp_db / "test2.h5", sources,
                   parallelism="future")

    check_db(db)
    check_feature_attrs(db.x)
    check_array_proxy(db.x)


def check_df_proxy(feat):
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
    check_df_proxy(db.x)

    db = DB.create(tmp_db / "test2.h5", sources,
                   parallelism="future")

    check_db(db)
    check_feature_attrs(db.x)
    check_df_proxy(db.x)


def check_dict_proxy(feat):
    check_array_proxy(feat.x)
    check_df_proxy(feat.y)
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
    check_dict_proxy(db.D)

    db = DB.create(tmp_db / "test1.h5", sources, parallelism='future')

    check_db(db)
    check_feature_attrs(db.D)
    check_dict_proxy(db.D)


def check_dictofdict_proxy(feat):
    check_dict_proxy(feat.p)
    check_dict_proxy(feat.q)
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
    check_dictofdict_proxy(db.D)

    db = DB.create(tmp_db / "test1.h5", sources, parallelism='future')

    check_db(db)
    check_feature_attrs(db.D)
    check_dictofdict_proxy(db.D)


#### DERIVED ARRAY


def test_derived_array(tmp_db):
    class DB(Database):
        x = Array()
        y = DerivedArray(derived_from="x")

    sources = tuple(map(str, range(20)))
    db = DB.create(tmp_db / "test1.h5", sources, parallelism='mp')

    check_db(db)
    check_feature_attrs(db.x)
    check_array_proxy(db.x)
    check_array_proxy(db.y)
    assert np.all(db.x.ids[()] == db.y.ids[()])
    # somehow they are not strictly equal...
    assert np.allclose(db.x[:], (db.y[:] - 1))


def check_after_feature(feat):
    assert np.all(feat[:] == 0)


# TODO : What does one want after load?
#  Build vocabularies?...
#  Change dtype, e.g. [0, 255] -> [0., 1.]?...
#  Normalize with mean and std?...


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


