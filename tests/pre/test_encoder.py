import pandas as pd
import pytest

# paso imports

from paso.base import Paso, PasoError
from paso.pre.encoders import Encoders

from loguru import logger

session = Paso(parameters_filepath="../../parameters/lesson.1.yaml").startup()

# 0
def test_Class_init_NoArg():
    with pytest.raises(PasoError):
        g = Encoders()


# 1
def test_Class_init_WrongScaler():
    with pytest.raises(PasoError):
        g = Encoders("GORG")


# BoxCoxScaler unit tests
# 2
def test_EncoderList(X):
    assert Encoders("BaseNEncoder").encoders() == [
        "BackwardDifferenceEncoder",
        "BinaryEncoder",
        "HashingEncoder",
        "HelmertEncoder",
        "OneHotEncoder",
        "OrdinalEncoder",
        "SumEncoder",
        "PolynomialEncoder",
        "BaseNEncoder",
        "LeaveOneOutEncoder",
        "TargetEncoder",
        "WOEEncoder",
        "MEstimateEncoder",
        "JamesSteinEncoder",
        "CatBoostEncoder",
        "EmbeddingVectorEncoder",
    ]


# 3
def test_bad_encoder_name():
    with pytest.raises(PasoError):
        g = Encoders("fred")


# 4
def test_BaseNEncoder_no_df(X):
    with pytest.raises(PasoError):
        Encoders(description_filepath="../../descriptions/pre/encoders/OHE.yaml").train(
            [["Male", 1], ["Female", 3], ["Female", 2]]
        )


# 5
def test_OrdinaEncoders(X):
    h = [["Male", 1], ["Female", 3], ["Female", 2]]
    hdf = pd.DataFrame(h)
    o = [[1, 1], [2, 3], [2, 2]]
    odf = pd.DataFrame(o)
    assert (
        (
            Encoders(description_filepath="../../descriptions/pre/encoders/OHE.yaml")
            .train(hdf)
            .predict(hdf)
            == odf
        )
        .any()
        .any()
    )


# 6
def test_OrdinaEncoderFlagsCacheOff(X):
    h = [["Male", 1], ["Female", 3], ["Female", 2]]
    hdf = pd.DataFrame(h)
    g = Encoders("OrdinalEncoder")
    g.train(hdf)
    assert (g.trained and not g.cache) == True


# 7
def test_OrdinaEncoderFlagsCacheOn(X):
    h = [["Male", 1], ["Female", 3], ["Female", 2]]
    hdf = pd.DataFrame(h)
    g = Encoders("OrdinalEncoder").cacheOn()
    g.train(hdf).predict(hdf)
    assert (g.trained and g.predicted and g.cache) == True


# 8
def test_OrdinaEncoderFlagsCacheOffpredictedNot(X):
    h = [["Male", 1], ["Female", 3], ["Female", 2]]
    hdf = pd.DataFrame(h)
    g = Encoders("OrdinalEncoder")
    g.train(hdf)
    assert (g.trained and not g.predicted and not g.cache) == True


# 9
def test_OrdinaEncoderLoadError():
    h = [["Male", 1], ["Female", 3], ["Female", 2]]
    hdf = pd.DataFrame(h)
    g = Encoders("OrdinalEncoder")
    g.train(hdf)
    with pytest.raises(PasoError):
        g.load()


# 10
def test_OrdinaEncoderSaveError():
    h = [["Male", 1], ["Female", 3], ["Female", 2]]
    hdf = pd.DataFrame(h)
    g = Encoders("OrdinalEncoder")
    g.train(hdf)
    with pytest.raises(PasoError):
        g.save()


# 11
def test_OrdinaEncoderFlagsWrite():
    h = [["Male", 1], ["Female", 3], ["Female", 2]]
    hdf = pd.DataFrame(h, columns=["a", "b"])
    g = Encoders("OrdinalEncoder").cacheOn()
    g.train(hdf).predict(hdf)
    fp: str = "tmp/df"
    assert g.write(fp) == g


# 12
def test_OrdinaEncoderFlagsWriteRead():
    h = [["Male", 1], ["Female", 3], ["Female", 2]]
    hdf = pd.DataFrame(h, columns=["a", "b"])
    g = Encoders("OrdinalEncoder").cacheOn()
    g.train(hdf)
    odf = g.predict(hdf)
    fp: str = "tmp/df"
    g.write(fp)
    assert (g.read(fp) == odf).any().any()


# 16
# 13

# 14

# 15


# 17


# 18

# 19
