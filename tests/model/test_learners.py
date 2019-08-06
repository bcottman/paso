from pathlib import Path
import pytest

# paso imports

from paso.base import Paso, Param, PasoError
from paso.learners.learners import Learner

fp = "../../parameters/default.yaml"
session = Paso(parameters_filepath=fp).startup()
# 0
def test_paso_param_file():
    assert session.parameters_filepath == fp


# 1
def test_paso_param_file_globa():
    assert session.parameters_filepath == Param.gfilepath


# 2
def test_sklearn_parm_global():
    assert Param.parameters_D["project"] == "Common Ground Solutions/paso"


# 3
def test_learn_train_ontological_file_not_exist(flower):
    o = Learner()
    with pytest.raises(PasoError):
        o.train(
            flower,
            ontological_filepath="../../ontologies/learners/RandomForestClassificationx.yaml",
        ) == o


# 4
def test_learn_train_bad_init(flower):
    o = Learner(
        ontological_filepath="../../ontologies/learners/RandomForestClassification.yaml"
    )
    with pytest.raises(PasoError):
        o.train(flower) == o


# 5
def test_learn_train_no_target(flower):
    o = Learner()
    with pytest.raises(PasoError):
        o.train(
            flower,
            ontological_filepath="../../ontologies/learners/RandomForestClassification.yaml",
        ) == o


# 6
def test_learn_train_kw_target(flower):
    o = Learner()
    o.train(
        flower,
        ontological_filepath="../../ontologies/learners/RandomForestClassification.yaml",
        target="TypeOf",
    )
    assert o.target == "TypeOf"


# 7


# 9

# 10
