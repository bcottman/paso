import warnings

warnings.filterwarnings("ignore")
import pytest

# paso imports
from paso.base import PasoError, Paso
from paso.pre.ContinuoustoCategory import ContinuoustoCategory

#
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"

session = Paso().startup("../../parameters/default-lesson.1.yaml")

# 1
def test_ContinuoustoCategory_no_passed_arg_type_error(df_type_low_V11):

    g = ContinuoustoCategory()
    with pytest.raises(PasoError):
        g.transform()


# 2
def test_ContinuoustoCategory_passed_arg_type_error(df_type_low_V11):
    g = ContinuoustoCategory()
    with pytest.raises(PasoError):
        g.transform(0)


# 3
def test_ContinuoustoCategory_transform_City_float(df_City):
    g = ContinuoustoCategory()
    CityC = df_City.copy()
    beforelen = 2 * len(CityC.columns)
    assert len(g.transform(CityC, inplace=True).columns) == beforelen


# 4
def test_ContinuoustoCategory_transform_City_ignore_integer(df_City):
    g = ContinuoustoCategory()
    dfi = df_City.astype("int")
    assert (g.transform(dfi, integer=False) == dfi).all().all()


# 4b
def test_ContinuoustoCategory_transform_City_ignore_integer_inplace_True(df_City):
    g = ContinuoustoCategory()
    dfi = df_City.astype("int")
    assert (g.transform(dfi, integer=False, inplace=True) == dfi).all().all()
    # 4c


def test_ContinuoustoCategory_transform_City_ignore_integer_inplace_Drop_True(df_City):
    g = ContinuoustoCategory()
    CityC = df_City.copy()
    CityC[["NOX"]] = CityC[["NOX"]] * 100  # (parts per 1 billion)
    dfi = CityC.astype("int")
    assert (g.transform(dfi, integer=False, inplace=True, drop=True) == dfi).all().all()
    # 5


def test_ContinuoustoCategory_transform_City_integer(df_City):
    g = ContinuoustoCategory()
    CityC = df_City.copy()
    CityC[["NOX"]] = CityC[["NOX"]] * 100  # (parts per 1 billion)
    dfi = CityC.astype("int")
    assert (
        (g.transform(dfi, inplace=True, drop=True) == dfi.astype("category"))
        .all()
        .all()
    )
    # 5b


def test_ContinuoustoCategory_transform_City_integer_inplace_True(df_City):
    g = ContinuoustoCategory()
    CityC = df_City.copy()
    CityC[["NOX"]] = CityC[["NOX"]] * 100  # (parts per 1 billion)
    dfi = CityC.astype("int")
    assert (
        (g.transform(dfi, inplace=True, drop=False) == dfi.astype("category"))
        .all()
        .all()
    )
    # 5c


def test_ContinuoustoCategory_transform_City_integer_inplace_Trueno_NOX_scale(df_City):
    g = ContinuoustoCategory()
    CityC = df_City.copy()
    #    CityC[['NOX']] = CityC[['NOX']] * 100  # (parts per 1 billion)
    dfi = CityC.astype("int")
    with pytest.raises(IndexError):
        assert (
            (g.transform(dfi, inplace=True, drop=False) == dfi.astype("category"))
            .all()
            .all()
        )


# 6
def test_ContinuoustoCategory_transform_City_integer_column_names(df_City):
    g = ContinuoustoCategory()
    CityC = df_City.copy()
    CityC[["NOX"]] = CityC[["NOX"]] * 100  # (parts per 1 billion)
    dfi = CityC.astype("int")
    assert (
        g.transform(dfi, inplace=True).columns == dfi.astype("category").columns
    ).all()


# 6b
def test_ContinuoustoCategory_transform_City_integer_column_names_inplace_def_false(
    df_City
):
    g = ContinuoustoCategory()
    CityC = df_City.copy()
    CityC[["NOX"]] = CityC[["NOX"]] * 100  # (parts per 1 billion)
    dfi = CityC.astype("int")
    with pytest.raises(ValueError):
        assert (g.transform(dfi).columns == dfi.astype("category").columns).all()


# 7
def test_ContinuoustoCategory_transform_Xobect_ignore_object(Xobject):
    g = ContinuoustoCategory()
    assert (g.transform(Xobject, integer=False, inplace=True) == Xobject).all().all()


# 8
def test_ContinuoustoCategory_transform_Xobject_not_datetime_format(Xobject):
    g = ContinuoustoCategory()
    assert (
        (g.transform(Xobject, inplace=True) == Xobject.astype("category")).all().all()
    )


# 9
def test_ContinuoustoCategory_transform_Xboolean_ignore_object(Xboolean):
    g = ContinuoustoCategory()
    assert (g.transform(Xboolean, integer=False) == Xboolean).all().all()


# 10
def test_ContinuoustoCategory_transform_Xboolean(Xboolean):
    g = ContinuoustoCategory()
    assert (
        (g.transform(Xboolean, inplace=True) == Xboolean.astype("category")).all().all()
    )


# 10b
def test_ContinuoustoCategory_transform_Xboolea_inplace_def_false(Xboolean):
    g = ContinuoustoCategory()
    assert (g.transform(Xboolean) == Xboolean.astype("category")).all().all()


# 11
def test_ContinuoustoCategory_transform_df_internet_traffic_ignore_object_and_integer(
    df_internet_traffic
):
    g = ContinuoustoCategory()
    assert (
        (g.transform(df_internet_traffic, inplace=True) == df_internet_traffic)
        .all()
        .all()
    )


# 12
def test_ContinuoustoCategory_transform_df_internet_traffic(df_internet_traffic):
    g = ContinuoustoCategory()
    with pytest.raises(ValueError):
        assert (
            (g.transform(df_internet_traffic) == df_internet_traffic.astype("category"))
            .all()
            .all()
        )


# 13
def test_ContinuoustoCategory_transform_df_internet_traffic_ignore_datetime_and_integer(
    df_City
):
    g = ContinuoustoCategory()
    assert (g.transform(df_City, integer=False, inplace=True) == df_City).all().all()


# 14
def test_ContinuoustoCategory_passed_no_save(df_type_low_V11):
    g = ContinuoustoCategory()
    with pytest.raises(AttributeError):
        g.save("foo")


# 15
def test_ContinuoustoCategory_passed_no_load(df_type_low_V11):
    g = ContinuoustoCategory()
    with pytest.raises(AttributeError):
        g.load("foo")


# 16
def test_ContinuoustoCategory_passed_cacheOff(df_type_low_V11):
    g = ContinuoustoCategory().cacheOff()
    assert not g.cache


# 17
def test_ContinuoustoCategory_passed_cacheOn(df_type_low_V11):
    g = ContinuoustoCategory().reset().cacheOn()
    assert not g.transformed and g.cache and not g.persisted


# 18  # BUG IN PARQUET??
def test_ContinuoustoCategory_writeV11(df_type_low_V11):
    g = ContinuoustoCategory().reset().cacheOn()
    g.transform(df_type_low_V11, inplace=True)
    fp: str = "tmp/df"
    with pytest.raises(ValueError):
        g.write(fp)


# 19
def test_ContinuoustoCategory_write_df_internet_traffic(df_City):
    g = ContinuoustoCategory().reset().cacheOn()
    g.transform(df_City, inplace=True)
    fp: str = "tmp/df"
    g.write(fp)
    assert g.transformed and g.cache and g.persisted and (g.save_file_name == fp)


########
