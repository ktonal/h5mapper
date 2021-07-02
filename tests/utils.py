import numpy as np
import pandas as pd
import pytest
from h5mapper import Array


@pytest.fixture
def tmp_db(tmp_path):
    root = (tmp_path / "dbs")
    root.mkdir()
    return root


class FeatWithAttrs(Array):
    __re__ = r".*"

    param1 = 18
    param2 = "hey"
    param3 = None

    @property
    def attrs(self):
        return dict(param1=self.param1, param2=self.param2, param3=self.param3)

    def custom_method(self, inputs):
        return np.zeros_like(inputs)


class RandnArray(FeatWithAttrs):

    def load(self, source):
        return np.random.randn(4, 12)


class IntVector(FeatWithAttrs):
    def load(self, source):
        return np.random.randint(0, 32, (1, 8, ))


class RandnArrayWithT(RandnArray):
    __t__ = (
        lambda arr: arr.flat[:],
    )


class DF(FeatWithAttrs):

    def load(self, source):
        return pd.DataFrame(np.random.randn(4, 12))


class Dict(FeatWithAttrs):

    def load(self, source):
        return dict(x=np.random.randn(4, 12),
                    y=np.random.randn(4, 12)
                    )


class DictofDict(FeatWithAttrs):

    def load(self, source):
        return dict(
            p=dict(x=np.random.randn(4, 12),
                   y=np.random.randn(4, 12)
                   ),
            q=dict(x=np.random.randn(4, 12),
                   y=np.random.randn(4, 12)
                   )
        )


class DerivedArray(FeatWithAttrs):

    def __init__(self, derived_from="x"):
        self.derived_from = derived_from

    def load(self, source):
        return source + 1


class AfterFeature(FeatWithAttrs):

    def load(self, source):
        return RandnArray().load(source)

    def after_create(self, db, feature_key):
        print(self, feature_key)
        feat = db.get_proxy(feature_key)
        # modify the feature in-place (file should be open for write)
        feat[:] = np.zeros_like(feat[:])
        return None


class AfterArray(FeatWithAttrs):
    def load(self, source):
        return None

    def after_create(self, db, feature_key):
        return RandnArray().load(None)


class AfterDF(FeatWithAttrs):
    def load(self, source):
        return None

    def after_create(self, db, feature_key):
        return DF().load(None)


class AfterDict(FeatWithAttrs):
    def load(self, source):
        return None

    def after_create(self, db, feature_key):
        return {"x": RandnArray().load(None), "y": DF().load(None)}


class NoneFeat(Array):

    def load(self, source):
        return None


class BadFeat(Array):

    def load(self, source):
        return "this should raise an Exception"