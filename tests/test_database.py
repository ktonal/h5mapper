import pytest

from h5m import *
from h5m import Database

from .utils import *


@pytest.fixture
def custom_h5(tmp_path):
    # some custom file
    h5f = h5py.File(tmp_path / "custom.h5", "w")
    g1 = h5f.create_group("g1")
    g2 = h5f.create_group("g2")
    g1.create_dataset("d1", data=np.random.randn(100, 100))
    g2.create_dataset("d2", data=np.random.randn(100, 100))
    h5f.flush()
    h5f.close()
    return tmp_path / "custom.h5"


def test_maps_custom_h5file(custom_h5):

    h5f = h5py.File(custom_h5, "r")
    db = Database(custom_h5, "r")
    assert hasattr(db, "g1") and isinstance(db.g1, Proxy)
    assert hasattr(db, "g2") and isinstance(db.g2, Proxy)
    assert hasattr(db.g1, "d1") and isinstance(db.g1.d1, Proxy)
    assert hasattr(db.g2, "d2") and isinstance(db.g2.d2, Proxy)

    assert np.all(db.g1.d1[()] == h5f["g1/d1"][()])
    assert np.all(db.g2.d2[()] == h5f["g2/d2"][()])

    h5f.close()


def test_handler(custom_h5):
    db = Database(custom_h5, "r")
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

    assert "Database" in repr(db)

    db.info()


def test_crud_api(custom_h5):
    db = Database(custom_h5, "r+")
    db.add("0", {
        # new feature
        "g3": {"d3": np.random.randn(1, 100)},

    })
    db.flush()
    got = db.get("0")
    assert "g3" in got
    assert isinstance(db.g3, Proxy)


def test_in_mem():
    db = in_mem(Database)
    assert bool(db.handler("h5py"))
    assert not os.path.isfile(db.h5_file)
    db.add("0", {"ds": np.random.rand(3, 4, 5)})
    assert isinstance(db.get("0")["ds"], np.ndarray)
    db.close()
    assert not db._f


def test_as_temp():
    path = as_temp("test.h5")
    h5f = h5py.File(path, "w")
    assert bool(h5f)
    h5f.create_dataset("ds", data=np.random.rand(3, 4, 5))
    h5f.flush()
    h5f.close()
    assert os.path.isfile(path)
    os.remove(path)
    os.removedirs(os.path.split(path)[0])
