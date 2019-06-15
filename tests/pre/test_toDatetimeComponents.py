from pathlib import Path
import pandas as pd

# import copy, random
import numpy as np
import pytest
from paso.base import pasoModel, PasoError, _Check_No_NA_Values, Paso
from paso.base import is_DataFrame, toDataFrame
from paso.pre.toDatetimeComponents import toDatetimeComponents

#
import os, sys
import warnings

warnings.filterwarnings("ignore")
session =  Paso().startup('../../parameters/default-lesson.1.yaml')

# 1
def test_df_toDatetimeComponents_Class_transform_WrongArgType():
    g = toDatetimeComponents()
    with pytest.raises(PasoError):
        g.transform([1, 2, 3])


# 2
def test_toDatetimeComponents_passed_All_Componentss_drop_Date_True_Default(
    df_small_no_NA, df_small_NFeatures, NComponentFeatures
):
    g = toDatetimeComponents()
    dt = df_small_no_NA.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature], exact=True, errors="ignore", infer_datetime_format=True
        )

    assert (
        g.transform(dt).shape[1]
        == (df_small_NFeatures - 4) * NComponentFeatures + df_small_NFeatures - 5
    )


# 3
def test_toDatetimeComponents_passed_All_Components_drop_Date_False(
    df_small_no_NA, df_small_NFeatures, NComponentFeatures
):
    g = toDatetimeComponents()
    dt = df_small_no_NA.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature], exact=True, errors="ignore", infer_datetime_format=True
        )
    assert (
        g.transform(dt, drop=False).shape[1]
        == (df_small_NFeatures - 4) * NComponentFeatures + df_small_NFeatures
    )


# 4
def test_toDatetimeComponents_with_NA_ValueError(df_small_NA):
    g = toDatetimeComponents()
    with pytest.raises(PasoError):
        g.transform(df_small_NA)


# 5
def test_toDatetimeComponents_passed_Year_Component_drop_Date_True_Default(
    df_small, df_small_NFeatures
):
    h = toDatetimeComponents()
    dt = df_small.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature], exact=True, errors="ignore", infer_datetime_format=True
        )
    with pytest.raises(PasoError):
        h.transform(dt, components=["Year"])


# 6
def test_toDatetimeComponents_passed_Year_Component_drop_Date_True_Elapsed_Default(
    df_small_no_NA, df_small_NFeatures
):
    h = toDatetimeComponents()
    dt = df_small_no_NA.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature], exact=True, errors="ignore", infer_datetime_format=True
        )
    assert (
        h.transform(dt, drop=True, components=["Year", "Elapsed"]).shape[1]
        == (df_small_NFeatures - 4) * 2 + df_small_NFeatures - 5
    )


# 7
def test_toDatetimeComponents_passed_Year_Component_drop_Date_False(
    df_small_no_NA, df_small_NFeatures
):
    dt = df_small_no_NA.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature], exact=True, errors="ignore", infer_datetime_format=True
        )
    h = toDatetimeComponents()
    assert (
        h.transform(dt, drop=False, components=["Year"]).shape[1]
        == (df_small_NFeatures - 4) * 1 + df_small_NFeatures
    )


# 8
def test_toDatetimeComponents_passed_DoY_Elapsed_IME_Components_drop_Date_True(
    df_small_no_NA, df_small_NFeatures
):
    j = toDatetimeComponents()
    dt = df_small_no_NA.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature], exact=True, errors="ignore", infer_datetime_format=True
        )
    assert (
        j.transform(dt, components=["Dayofyear", "Elapsed", "Is_month_end"]).shape[1]
        == (df_small_NFeatures - 4) * 3 + df_small_NFeatures - 5
    )


# 9
def test_toDatetimeComponents_passed_DoY_Elapsed_IME_Components_drop_Date_False(
    df_small_no_NA, df_small_NFeatures
):
    j = toDatetimeComponents()
    dt = df_small_no_NA.copy()
    for feature in dt.columns[0:6]:
        dt[feature] = pd.to_datetime(
            dt[feature], exact=True, errors="ignore", infer_datetime_format=True
        )
    assert (
        j.transform(
            dt, drop=False, components=["DayofYear", "Elapsed", "Is_month_end"]
        ).shape[1]
        == (df_small_NFeatures - 4) * 3 + df_small_NFeatures
    )


# 10
def test_df_toDatetimeComponents_write_bad():
    g = toDatetimeComponents()
    with pytest.raises(PasoError):
        g.write()


# 11
def test_df_toDatetimeComponents_write(df_small_no_NA):
    g = toDatetimeComponents().cacheOn()
    xxx = g.transform(df_small_no_NA, inplace=False)
    assert g.write("tmp/test.txt")


# 12
def test_df_toDatetimeComponents_read_bad_path():
    g = toDatetimeComponents()
    with pytest.raises(PasoError):
        g.read()


# 13
def test_df_toDatetimeComponents_read(df_small_no_NA):
    g = toDatetimeComponents().cacheOn()
    xxx = g.transform(df_small_no_NA)
    g.write("tmp/test.txt")
    assert (g.read("tmp/test.txt") == xxx).all().all()


# 14
def test_toDatetimeComponents_transform_df_internet_traffic_datetime(
    df_internet_traffic
):
    dt = df_internet_traffic.copy()
    dt["date"] = pd.to_datetime(
        df_internet_traffic["date"],
        exact=True,
        errors="ignore",
        infer_datetime_format=True,
    )
    g = toDatetimeComponents().cacheOn()
    assert (
        g.transform(dt, drop=True).columns
        == [
            "bit",
            "byte",
            "Year",
            "Month",
            "Week",
            "Day",
            "Dayofweek",
            "Dayofyear",
            "Elapsed",
            "Is_month_end",
            "Is_month_start",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
    )


# 14
def test_toDatetimeComponents_transform_df_internet_traffic_datetime(
    df_internet_traffic
):
    dt = df_internet_traffic.copy()
    dt["date"] = pd.to_datetime(
        df_internet_traffic["date"],
        exact=True,
        errors="ignore",
        infer_datetime_format=True,
    )
    g = toDatetimeComponents().cacheOn()
    assert (
        g.transform(dt, drop=True).columns
        == [
            "bit",
            "byte",
            "Year",
            "Month",
            "Week",
            "Day",
            "Dayofweek",
            "Dayofyear",
            "Elapsed",
            "Is_month_end",
            "Is_month_start",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
    )


# 15
def test_toDatetimeComponents_transform_df_internet_traffic_datetime_no_drop(
    df_internet_traffic
):
    dt = df_internet_traffic.copy()
    dt["date"] = pd.to_datetime(
        df_internet_traffic["date"],
        exact=True,
        errors="ignore",
        infer_datetime_format=True,
    )
    g = toDatetimeComponents().cacheOn()
    assert (
        g.transform(dt, drop=False).columns
        == [
            "date",
            "bit",
            "byte",
            "Year",
            "Month",
            "Week",
            "Day",
            "Dayofweek",
            "Dayofyear",
            "Elapsed",
            "Is_month_end",
            "Is_month_start",
            "Is_quarter_end",
            "Is_quarter_start",
            "Is_year_end",
            "Is_year_start",
        ],
    )
