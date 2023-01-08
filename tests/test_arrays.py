import pytest

from h5mapper import *
from .utils import *


@pytest.fixture
def array_db(tmp_db):

    class DB(TypedFile):
        x = RandnArray()

    sources = tuple(map(str, range(8)))
    db = DB.create(tmp_db / "test1.h5", sources,
                   parallelism="mp")
    return db


def check_handler(feat, keep_open):
    if keep_open:
        assert feat.owner.f_
    else:
        assert not feat.owner.f_


def test_proxy_attributes(array_db):
    feat = array_db.x

    assert feat.owner is array_db
    assert not feat.is_group
    assert hasattr(feat, "shape") and isinstance(feat.shape, tuple)

    # Proxies that should be hosted by the feature
    # assert hasattr(feat, 'src') and isinstance(feat.src, Proxy)
    assert hasattr(feat, 'refs') and isinstance(feat.refs, Proxy)

    # methods that should be available to the proxy
    assert feat.custom_method is not None
    invoked = feat.custom_method(np.ones((3, 4, 5)))
    assert np.all(invoked == np.zeros((3, 4, 5)))

    # check that keeping the db open issues only one handler
    feat.owner.keep_open = True
    hf = feat.handle()
    assert hf is feat.owner.handle('r')
    # should now be opened
    assert isinstance(feat[:], np.ndarray)
    assert hf is feat.owner.handle('r')
    hf.close()
    assert not feat.owner.f_
    feat.owner.keep_open = False


@pytest.mark.parametrize("keep_open", [True, False])
def test_proxy_item_api(array_db, keep_open):
    array_db.keep_open = keep_open

    feat = array_db.x
    # handler should be closed before first request
    check_handler(feat, False)

    # GET item
    # since ids are strings, they should be wrapped in .asstr()
    assert isinstance(feat[:], np.ndarray)
    check_handler(feat, keep_open)
    # assert feat.ids.asstr and isinstance(feat.ids[0], str)

    assert isinstance(feat[feat.refs[0]], np.ndarray)
    assert feat[feat.refs[0]].shape == RandnArray().load(None).shape
    check_handler(feat, keep_open)

    # SET item
    # single int
    before = feat[0]
    feat[0] = before + 1
    check_handler(feat, keep_open)
    assert np.allclose(feat[0], before + 1)
    # slice
    before = feat[:5]
    feat[:5] = before + 1
    check_handler(feat, keep_open)
    assert np.allclose(feat[:5], before + 1)
    check_handler(feat, keep_open)
    # arrays
    indices = np.array([0, 2, 5, 12])
    before = feat[indices]
    feat[indices] = before + 1
    check_handler(feat, keep_open)
    assert np.allclose(feat[indices], before + 1)
    check_handler(feat, keep_open)


@pytest.mark.parametrize("keep_open", [True, False])
def test_proxy_source_api(array_db, keep_open):
    array_db.keep_open = keep_open

    feat = array_db.x
    # handler should be closed before first request
    check_handler(feat, False)

    # source-wise GET and SET

    new_arr = RandnArray().load(None)
    feat.add("21", new_arr)
    check_handler(feat, keep_open)
    assert np.allclose(feat.get("21"), new_arr)
    check_handler(feat, keep_open)
    feat.set("21", new_arr + 2.)
    check_handler(feat, keep_open)
    assert np.allclose(feat.get("21"), new_arr + 2.)
    check_handler(feat, keep_open)



