import warnings

warnings.filterwarnings("ignore")
import pytest

# paso imports
from paso.base import PasoError,Paso
from paso.pre.toCategory import toCategory

#
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
session =  Paso().startup('../../parameters/default-lesson.1.yaml')

# 1
def test_toCategory_no_passed_arg_type_error(df_type_low_V11):
    g = toCategory()
    with pytest.raises(PasoError):
        g.transform()


# 2
def test_toCategory_passed_arg_type_error(df_type_low_V11):
    g = toCategory()
    with pytest.raises(PasoError):
        g.transform(0)


# 3
def test_toCategory_transform_City_float(df_City):
    g = toCategory()
    assert (g.transform(df_City) == df_City).all().all()


# 4
def test_toCategory_transform_City_ignore_integer(df_City):
    g = toCategory()
    dfi = df_City.astype("int")
    assert (
        (g.transform(dfi, integer=False, boolean=False, objects=False) == dfi)
        .all()
        .all()
    )


# 5
def test_toCategory_transform_City_integer(df_City):
    g = toCategory()
    dfi = df_City.astype("int")
    assert (g.transform(dfi) == dfi.astype("category")).all().all()


# 6
def test_toCategory_transform_City_integer_column_names(df_City):
    g = toCategory()
    dfi = df_City.astype("int")
    assert (g.transform(dfi).columns == dfi.astype("category").columns).all()


# 7
def test_toCategory_transform_Xobect_ignore_object(Xobject):
    g = toCategory()
    assert (
        (g.transform(Xobject, integer=False, boolean=False, objects=False) == Xobject)
        .all()
        .all()
    )


# 8
def test_toCategory_transform_Xobject_not_datetime_format(Xobject):
    g = toCategory()
    assert (g.transform(Xobject) == Xobject.astype("category")).all().all()


# 9
def test_toCategory_transform_Xboolean_ignore_object(Xboolean):
    g = toCategory()
    assert (
        (g.transform(Xboolean, integer=False, boolean=False, objects=False) == Xboolean)
        .all()
        .all()
    )


# 10
def test_toCategory_transform_Xboolean(Xboolean):
    g = toCategory()
    assert (g.transform(Xboolean) == Xboolean.astype("category")).all().all()


# 11
def test_toCategory_transform_df_internet_traffic_ignore_object_and_integer(
    df_internet_traffic
):
    g = toCategory()
    assert (
        (
            g.transform(
                df_internet_traffic, integer=False, boolean=False, objects=False
            )
            == df_internet_traffic
        )
        .all()
        .all()
    )


# 12
def test_toCategory_transform_df_internet_traffic(df_internet_traffic):
    g = toCategory()
    assert (
        (g.transform(df_internet_traffic) == df_internet_traffic.astype("category"))
        .all()
        .all()
    )


# 13
def test_toCategory_transform_df_internet_traffic_ignore_datetime_and_integer(
    df_internet_traffic
):
    g = toCategory()
    assert (
        (
            g.transform(
                df_internet_traffic, integer=False, boolean=False, objects=False
            )
            == df_internet_traffic
        )
        .all()
        .all()
    )


# 14
def test_toCategory_passed_no_save(df_type_low_V11):
    g = toCategory()
    with pytest.raises(AttributeError):
        g.save("foo")


# 15
def test_toCategory_passed_no_load(df_type_low_V11):
    g = toCategory()
    with pytest.raises(AttributeError):
        g.load("foo")


# 16
def test_toCategory_passed_cacheOff(df_type_low_V11):
    g = toCategory().cacheOff()
    assert not g.cache


# 17
def test_toCategory_passed_cacheOn(df_type_low_V11):
    g = toCategory().reset().cacheOn()
    assert not g.transformed and g.cache and not g.persisted


# 18  # BUG IN PARQUET??
def test_toCategory_writeV11(df_type_low_V11):
    g = toCategory().reset().cacheOn()
    g.transform(df_type_low_V11, inplace=True)
    fp: str = "tmp/df"
    with pytest.raises(ValueError):
        g.write(fp)


# 19
def test_toCategory_write_df_internet_traffic(df_internet_traffic):
    g = toCategory().reset().cacheOn()
    g.transform(df_internet_traffic, inplace=True)
    fp: str = "tmp/df"
    g.write(fp)
    assert g.transformed and g.cache and g.persisted and (g.save_file_name == fp)


########
