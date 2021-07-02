import pytest

from h5mapper import *
from h5mapper.crud import _load
from .utils import *


def test_load():
    pass


@pytest.mark.parametrize("para", ["mp", "future", "other"])
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

