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

from paso.pre.cleaners import Class_Balance,Augment_by_Class

#
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
session =  Paso(parameters_filepath='../../parameters/lesson.1.yaml').startup()

# 1
def test_Remove_Features_no_passed_arg_type_error(City):
    g = Remove_Features()
    with pytest.raises(PasoError):
        g.transform()
# 2
def test_Remove_Features_passed_arg_type_error(City):
    g = Remove_Features()
    with pytest.raises(KeyError):
        g.transform(City,unplace=False,verbose=False,remove=0)
# 3
def test_Remove_Features_none(City):
    g = Remove_Features()
    assert (g.transform(City,inplace=True,remove=[]) ==City).all().all()
# 4
def test_Remove_Features(City):
    g = Remove_Features()
    City['bf'] = 999
    City['bf2'] = 2
    assert (g.transform(City,inplace=True,remove=['bf','bf2']) == City).all().all()
# 5
def test_Class_Balanced_classBalancers():
    assert Class_Balance().classBalancers() == ['RanZOverSample',
                                 'SMOTE',
                                 'ADASYN',
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
def test_Class_Balanced_noClassTargetss(City):
    o = Class_Balance()
    with pytest.raises(PasoError):
        o.transform(City[City.columns.difference(['MEDV'])],City['MEDV'])
# 7
def test_Class_Balanced_Smote_Flower(flower):
    o = Class_Balance(ontological_filepath='../../ontologies/pre/cleaners/SMOTE.yaml')
    assert o.transform(flower,'TypeOf').shape == (150,5)
#8
def test_Augment_by_Class_no_Ratio(flower):
    o = Augment_by_Class(ontological_filepath='../../ontologies/pre/cleaners/SMOTE.yaml')
    with pytest.raises(PasoError):
        assert o.transform(flower,'TypeOf').shape == (150,5)
#9
def test_Augment_by_Class_Smote_Flower(flower):
    o = Augment_by_Class(ontological_filepath='../../ontologies/pre/cleaners/SMOTE.yaml')
    assert o.transform(flower,'TypeOf',1.0).shape == (300,5)

# 8
# def test_Class_Balanced_noClassTargetss(City):
#     o = Class_Balance(ontological_filepath='../../ontologies/pre/cleaners/SMOTENC.yaml')
#     assert o.transform(City,'MEDV').shape == (506,14)


########
