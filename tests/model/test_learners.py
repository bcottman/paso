from pathlib import Path

import pytest

# paso imports

from paso.base import Paso, Param
from paso.models.learners import Learner

fp = "../../parameters/test_sklearn.yaml"
session = Paso(parameters_filepath=fp).startup()
#0
def test_paso_param_file():
    assert session.parameters_filepath == fp
#1
def test_paso_param_file_globa():
    assert session.parameters_filepath == Param.filepath
# 2
def test_learn_parm_global():
    assert Param.parameters_D['project'] == 'Common Ground Solutions/paso'
#3
def test_learn_train_bad_target(flower):
    m = Learner('RandomForest', 'Classifier',)
    assert m.train(flower) == m
