import warnings

warnings.filterwarnings("ignore")
import pytest
import pandas as pd

# paso imports
from paso.base import PasoError,Paso
from paso.pre.cleaners import Features_not_in_train_or_test\
    ,Dupilicate_Features_by_Values\
    ,Features_with_Single_Unique_Value\
    ,Features_Variances\
    ,Remove_Features

#
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
session =  Paso(parameters_filepath='../../parameters/default-lesson.1.yaml').startup()

# 1
def test_Remove_Features_no_passed_arg_type_error(df_City):
    g = Remove_Features()
    with pytest.raises(PasoError):
        g.transform()


# 2
def test_Remove_Features_passed_arg_type_error(df_City):
    g = Remove_Features()
    with pytest.raises(KeyError):
        g.transform(df_City,inplace=False,remove=0)


# 3
def test_Remove_Features_none(df_City):
    g = Remove_Features()
    assert (g.transform(df_City,inplace=True,remove=[]) == df_City).all().all()


# 4
def test_Remove_Features_none(df_City):
    g = Remove_Features()
    df_City['bf'] = 999
    df_City['bf2'] = 2
    assert (g.transform(df_City,inplace=True,remove=['bf','bf2']) == df_City).all().all()


# 5
from paso.pre.cleaners import Class_Balance
def test_Class_Balanced_classBalancers():
    o = Class_Balance('SMOTENC', categorical_features=[])
    assert o.classBalancers() == ['RanZOverSample',
                                 'SMOTE',
                                 'ADASYN:',
                                 'BorderLineSMOTE',
                                 'SVMSMOTE',
                                 'SMOTENC',
                                 'RandomUnderSample',
                                 'ClusterCentroids',
                                 'NearMiss',
                                 'EditedNearestNeighbour',
                                 'RepeatedEditedNearestNeighbours',
                                 'CondensedNearestNeighbour',
                                 'OneSidedSelection']
# 6

def test_Class_Balanced_noClassTargetss(df_City):
    o = Class_Balance('SMOTENC', categorical_features=[])
    with pytest.raises(IndexError):
        o.transform(df_City[df_City.columns.difference(['MEDV'])],df_City['MEDV'])
# 7



# 8






########
