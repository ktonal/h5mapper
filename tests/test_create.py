import pytest

from h5mapper import *
from h5mapper.create import _compute
from h5mapper.crud import _load
from .utils import *
import pickle


def test_load():
    pass


@pytest.mark.parametrize("para", ["mp", "threads", 'none', "other"])
def test_parallelism(tmp_path, para):
    class DB(TypedFile):
        x = RandnArray()

    sources = tuple(map(str, range(8)))
    if para != "other":
        db = DB.create(tmp_path / "test1.h5", sources,
                       parallelism=para)
        assert isinstance(db, TypedFile)
    else:
        with pytest.raises(ValueError):
            db = DB.create(tmp_path / "test1.h5", sources,
                           parallelism=para)


@pytest.mark.parametrize('schema', [None, {}, {"x": RandnArray()}])
def test_schemas(tmp_path, schema):
    if schema is None:
        class DB(TypedFile):
            x = RandnArray()

        sources = tuple(map(str, range(8)))
        db = DB.create(tmp_path / "test1.h5", sources,
                       parallelism="mp")
        assert isinstance(db, TypedFile)
    elif schema == {}:
        sources = tuple(map(str, range(8)))
        with pytest.raises(ValueError):
            db = TypedFile.create(tmp_path / "test1.h5", sources,
                                  schema=schema,
                                  parallelism="mp")
    else:
        sources = tuple(map(str, range(8)))
        db = TypedFile.create(tmp_path / "test1.h5", sources,
                              schema=schema,
                              parallelism="mp")
        assert isinstance(db, TypedFile)
        assert isinstance(db.x[:], np.ndarray)


def add1(x):
    return x + 1


class DB(TypedFile):
    x = RandnArray()
    z = RandnArray()


@pytest.mark.parametrize("para", ["mp", 'none', 'threads'])
def test_compute(tmp_path, para):
    sources = tuple(map(str, range(8)))
    DB.create(tmp_path / (para + "-test1.h5"), sources, parallelism="none",
              mode='w',
              keep_open=False)
    db = DB(tmp_path / (para + "-test1.h5"), mode='r+', keep_open=False)
    assert isinstance(db, TypedFile)
    n_prior_ids = len(sources)
    # apply_and_store({'y': add1}, db.x, '0')
    _compute({'y': lambda x: x+1}, db.x, para, 1, db)

    assert isinstance(db.y, Proxy)
    # no new id is created
    assert len(db.index) == n_prior_ids
    # values stored are correct
    assert ((db.x[:] + 1) == db.y[:]).all()

    # all features have all the refs
    assert len(db.y.refs[:]) == n_prior_ids
    assert len(db.x.refs[:]) == n_prior_ids
    assert len(db.z.refs[:]) == n_prior_ids
