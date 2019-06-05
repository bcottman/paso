import pytest
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

# paso imports
from paso.base import pasoError,Paso
from paso.pre.EliminateUnviableFeatures import EliminateUnviableFeatures

__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
session =  Paso().startup('../../parameters/default-lesson.1.yaml')

#1
@pytest.mark.smoke
def test_set_no_argsdf_City(df_City):
    g = EliminateUnviableFeatures(verbose=True)
    assert (g.transform(df_City).columns == df_City.columns).any()
#2
@pytest.mark.smoke
def test_set_ignore_arg_error(df_City):
    with pytest.raises(TypeError):
        g = EliminateUnviableFeatures(ignore=[1, 14, 'foo'])
#3
@pytest.mark.smoke
def test_df_EliminateUnviableFeatures_Class_init_WrongArg():
    with pytest.raises(TypeError):
        g = EliminateUnviableFeatures(Bad='bad')
#4
@pytest.mark.smoke
def test_set_ignore_arg(df_City):
    g = EliminateUnviableFeatures()
    g.transform(df_City,ignore = [1, 14, 'foo'])
    assert g.ignore == [1, 14, 'foo']
#5fix
@pytest.mark.smoke
def test_df_EliminateUnviableFeatures_passed_arg_type_error(df_City):
    g = EliminateUnviableFeatures()
    assert type(g.transform(df_City))  == type(pd.DataFrame())
#6
@pytest.mark.smoke
def test_df_EliminateUnviableFeatures_with_NA_Values(df_typeNA):
    g = EliminateUnviableFeatures()
    with pytest.raises(pasoError):
        g.transform(df_typeNA)
#7
@pytest.mark.smoke
def test_df_EliminateUnviableFeatures_Duplicate_feature_values(df_typeDup):
    g = EliminateUnviableFeatures()
    assert(len(g.transform(df_typeDup.copy()).columns) == (len(df_typeDup.columns) - 1))
#8
@pytest.mark.smoke
def test_df_EliminateUnviableFeatures_Orthognal(df_type,df_typeo):
    g = EliminateUnviableFeatures()
    with pytest.raises(IndexError):
        g.transform(df_type, Yarg=df_typeo,inplace=True).iloc[0,0]
#9
@pytest.mark.smoke
def test_df_EliminateUnviableFeatures_Identical(df_type, df_typeo):
    g = EliminateUnviableFeatures()
    assert (g.transform(df_type, Yarg=df_type, inplace=True) == df_type).all().all()
#10

#11
def test_df_EliminateUnviableFeatures_Orthognal_City_paso_Error(df_type,df_City):
    g = EliminateUnviableFeatures()
    with pytest.raises(ValueError):
        assert (g.transform(df_type, Yarg=df_City) == df_type).all().all()
#12
@pytest.mark.smoke
def test_df_EliminateUnviableFeatures_FitToSelf():
    g = EliminateUnviableFeatures()
    with pytest.raises(AttributeError):
        (g.fit() == g)
#13
@pytest.mark.smoke
def test_Eliminate_Single_Unique_Value_Features(df_type_SV):
    print(df_type_SV)
    g = EliminateUnviableFeatures()
    assert (len(g.transform(df_type_SV.copy()).columns) == (len(df_type_SV.columns)  -1))
#14
@pytest.mark.smoke
def test_Eliminate_Low_Variance_Features_LE_10_values(df_type_low_V):
    g = EliminateUnviableFeatures()
    assert (len(g.transform(df_type_low_V).columns) == len(df_type_low_V.columns))
#15
@pytest.mark.smoke
def test_set_SD_LIMIT(df_City):
    sdev = 0.356
    g = EliminateUnviableFeatures()
    g.transform(df_City,SD_LIMIT=sdev)
    assert (g.SD_LIMIT == sdev)
#16
def test_Eliminate_Low_Variance_Features_passed_SD(df_type_low_V11):
    sdev = 0.3
    g = EliminateUnviableFeatures()
    ll = len(g.transform(df_type_low_V11.copy(),inplace=False,SD_LIMIT=sdev).columns)
    assert (ll == len(df_type_low_V11.columns)-1 )
