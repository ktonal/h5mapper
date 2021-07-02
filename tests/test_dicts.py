import pytest

from h5mapper import *
from .utils import *


@pytest.fixture
def dict_db(tmp_db):

    class DB(TypedFile):
        d = Dict()

    sources = tuple(map(str, range(8)))
    db = DB.create(tmp_db / "test1.h5", sources,
                   parallelism="mp")
    return db


def check_handler(feat, keep_open):
    if keep_open:
        assert feat.owner.f_
    else:
        assert not feat.owner.f_


def test_proxy_attributes(dict_db):
    feat = dict_db.d

    assert feat.attrs == FeatWithAttrs().attrs

    assert feat.owner is dict_db
    assert feat.is_group

    # Proxies that should be hosted by the feature
    assert hasattr(feat, 'x') and isinstance(feat.x, Proxy)
    assert hasattr(feat, 'y') and isinstance(feat.y, Proxy)

    # methods that should be available to the proxy
    assert feat.custom_method is not None
    invoked = feat.custom_method(np.ones((3, 4, 5)))
    assert np.all(invoked == np.zeros((3, 4, 5)))

    # check that keeping the db open issues only one handler
    feat.owner.keep_open = True
    hf = feat.handle()
    assert hf is feat.owner.handle('r')
    # should now be opened
    assert isinstance(feat["0"], dict)
    assert hf is feat.owner.handle('r')
    hf.close()
    assert not feat.owner.f_
    feat.owner.keep_open = False


@pytest.mark.parametrize("keep_open", [True, False])
def test_proxy_api(dict_db, keep_open):
    dict_db.keep_open = keep_open

    feat = dict_db.d
    # handler should be closed before first request
    check_handler(feat, False)

    # GET item == get(src)
    assert isinstance(feat["0"], dict)
    check_handler(feat, keep_open)
    got1, got2 = feat["0"], feat.get("0")
    assert np.all(got1["x"] == got2["x"])
    assert np.all(got1["y"] == got2["y"])
    check_handler(feat, keep_open)

    # NOT SUPPORTED TYPES
    with pytest.raises(TypeError):
        null = feat[0]
    with pytest.raises(TypeError):
        feat[0:8] = {}
    with pytest.raises(TypeError):
        null = feat[["3", "5", "7"]]

    # since ids are strings, they should be wrapped in .asstr()
    # assert feat.ids.asstr and isinstance(feat.ids[0], str)

    # SET item
    before = feat["0"]
    before.update({"x": before["x"] + 1})
    feat["0"] = before
    check_handler(feat, keep_open)
    assert np.allclose(feat["0"]["x"], before["x"])

    before = feat["0"]
    before.update({"x": before["x"] + 1})
    feat.set("0", before)
    check_handler(feat, keep_open)
    assert np.allclose(feat["0"]["x"], before["x"])

    # set new id

    before = feat["0"]
    before.update({"x": before["x"] + 1})
    feat.set("10", before)
    check_handler(feat, keep_open)
    assert np.allclose(feat["10"]["x"], before["x"])

    # ADD item
    val = Dict().load(None)
    feat.add("8", val)
    check_handler(feat, keep_open)
    assert np.all(feat["8"]["x"] == val["x"])
    assert np.all(feat["8"]["y"] == val["y"])

    # Add item with new child feature

    val = Dict().load(None)
    val["z"] = val.pop("y")
    feat.add("9", val)
    check_handler(feat, keep_open)
    dict_db.info()
    assert np.all(feat["9"]["x"] == val["x"])
    assert np.all(feat["9"]["z"] == val["z"])
    check_handler(feat, keep_open)
    assert hasattr(feat, "z") and isinstance(feat.z, Proxy)
    check_handler(feat, keep_open)


@pytest.fixture
def dict_of_dict_db(tmp_db):

    class DB(TypedFile):
        d = DictofDict()

    sources = tuple(map(str, range(8)))
    db = DB.create(tmp_db / "test1.h5", sources,
                   parallelism="mp")
    return db


def test_dictofdict_attributes(dict_of_dict_db):
    feat = dict_of_dict_db.d

    assert feat.attrs == FeatWithAttrs().attrs

    assert feat.owner is dict_of_dict_db
    assert feat.is_group

    # Proxies that should be hosted by the feature
    # assert hasattr(feat, 'src') and isinstance(feat.src, Proxy)
    # assert hasattr(feat, 'refs') and isinstance(feat.refs, Proxy)
    # assert hasattr(feat, 'ids') and isinstance(feat.ids, Proxy)
    assert hasattr(feat, 'p') and isinstance(feat.p, Proxy)
    assert hasattr(feat, 'q') and isinstance(feat.q, Proxy)
    assert hasattr(feat.p, 'x') and isinstance(feat.p.x, Proxy)
    assert hasattr(feat.q, 'y') and isinstance(feat.q.y, Proxy)


@pytest.mark.parametrize("keep_open", [True, False])
def test_dictofdict_api(dict_of_dict_db, keep_open):
    dict_of_dict_db.keep_open = keep_open

    feat = dict_of_dict_db.d
    # handler should be closed before first request
    check_handler(feat, False)

    # GET item == get(src)
    assert isinstance(feat["0"], dict)
    check_handler(feat, keep_open)
    got1, got2 = feat["0"], feat.get("0")
    assert np.all(got1["p"]["x"] == got2["p"]["x"])
    assert np.all(got1["q"]["y"] == got2["q"]["y"])
    check_handler(feat, keep_open)

    # NOT SUPPORTED TYPES
    with pytest.raises(TypeError):
        null = feat[0]
    with pytest.raises(TypeError):
        null = feat[0:8]
    with pytest.raises(TypeError):
        null = feat[["3", "5", "7"]]

    # since ids are strings, they should be wrapped in .asstr()
    # assert feat.ids.asstr and isinstance(feat.ids[0], str)

    # SET item
    before = feat["0"]
    before["p"].update({"x": before["p"]["x"] + 1})
    feat["0"] = before
    check_handler(feat, keep_open)
    assert np.allclose(feat["0"]["p"]["x"], before["p"]["x"])

    before = feat["0"]
    before["p"].update({"x": before["p"]["x"] + 1})
    feat.set("0", before)
    check_handler(feat, keep_open)
    assert np.allclose(feat["0"]["p"]["x"], before["p"]["x"])

    # ADD item
    val = DictofDict().load(None)
    feat.add("8", val)
    check_handler(feat, keep_open)
    assert np.all(feat["8"]["p"]["x"] == val["p"]["x"])
    assert np.all(feat["8"]["q"]["y"] == val["q"]["y"])

    # Add item with new child feature

    val = Dict().load(None)
    val["z"] = val.pop("y")
    feat.add("9", val)
    check_handler(feat, keep_open)
    assert np.all(feat["9"]["x"] == val["x"])
    assert np.all(feat["9"]["z"] == val["z"])
    check_handler(feat, keep_open)
    assert hasattr(feat, "z") and isinstance(feat.z, Proxy)
    check_handler(feat, keep_open)