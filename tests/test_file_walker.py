import pytest
import re
import os


from h5m.file_walker import FileWalker


def test_find_matches(tmp_path):
    (tmp_path / "sub").mkdir()
    files = [
        str(tmp_path / d / f)
        for d in ["", "sub"]
        for f in ["a.ext", "a.null", "b.ext", "b.null",
                  ".match-but-hidden.ext"]
    ]
    for f in files:
        open(f, "w").write(" ")
        assert os.path.isfile(f)

    # finds single file
    fw = FileWalker(r"ext$", str(tmp_path / "a.ext"))
    found = list(fw)
    assert len(found) == 1
    assert all(re.search(r"ext$", f) for f in found)

    # finds files
    fw = FileWalker(r"ext$", str(tmp_path))
    found = list(fw)
    assert len(found) == 4
    assert all(re.search(r"ext$", f) for f in found)
    assert len([f for f in found if "sub" in f]) == 2

    # finds directories
    fw = FileWalker(r"sub", str(tmp_path))
    found = list(fw)
    assert len(found) == 4
    assert all(re.search(r"sub", f) for f in found)
    assert len([f for f in found if ".ext" in f]) == 2

    # finds both
    fw = FileWalker(r"ext$", [str(tmp_path / "sub"), str(tmp_path / "a.ext"), str(tmp_path / "b.ext")])
    found = list(fw)
    assert len(found) == 4
    assert all(re.search(r"ext$", f) for f in found)
    assert len([f for f in found if "sub" in f]) == 2

    # test multiple iter()
    assert list(fw) == found


def test_raises_on_filenotfound(tmp_path):
    (tmp_path / "sub").mkdir()
    files = [
        str(tmp_path / d / f)
        for d in ["", "sub"]
        for f in ["a.ext", "a.null", "b.ext", "b.null"]
    ]
    for f in files:
        open(f, "w").write(" ")
        assert os.path.isfile(f)

    faulty_path = str(tmp_path / "subext")
    fw = FileWalker(r"ext", faulty_path)
    assert not os.path.exists(faulty_path)
    with pytest.raises(FileNotFoundError):
        found = list(fw)

    # also in lists
    fw = FileWalker(r"ext", [faulty_path])
    assert not os.path.exists(faulty_path)
    with pytest.raises(FileNotFoundError):
        found = list(fw)