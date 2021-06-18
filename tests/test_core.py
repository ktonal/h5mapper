from h5m import Database


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