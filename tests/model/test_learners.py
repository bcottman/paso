from pathlib import Path
import pytest
from loguru import logger

# paso imports

from paso.base import Paso, Param, PasoError
from paso.pre.inputers import Inputer
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
    o = Learner(
        ontological_filepath="../../ontologies/learners/RandomForestClassificationx.yaml"
    )
    with pytest.raises(PasoError):
        o.train(flower) == o


# 4
def test_learn_train_bad_init(flower):
    o = Learner(
        ontological_filepath="../../ontologies/learners/RandomForestClassification.sm.yaml"
    )
    with pytest.raises(PasoError):
        o.train(flower) == o


# 5
def test_learn_train_no_target(flower):
    o = Learner(
        ontological_filepath="../../ontologies/learners/RandomForestClassification.sm.yaml"
    )
    with pytest.raises(PasoError):
        o.train(flower) == o


# 6
def test_learn_train_kw_target_iris(flower):
    o = Learner(
        ontological_filepath="../../ontologies/learners/RFC.sm.sm.yaml"
    )
    o.train(flower, target="TypeOf")
    o.predict(flower, measure=True)
    logger.debug(o.metrics)
    assert o.target == "TypeOf"


# 7
def test_learn_train_kw_target_pima():
    inputer = Inputer(
        ontological_filepath="../../ontologies/pre/inputers/pima-diabetes.yaml"
    )
    diabetes = inputer.transform()
    leaner = Learner(
        ontological_filepath="../../ontologies/learners/RFC.sm.yaml"
    )
    leaner.train(
        diabetes, target=inputer.target, checkpoint="diabetesRandomForest1.ckp"
    )
    assert leaner.target == inputer.target


# 8

def test_learn_predict_metrics():
    inputer = Inputer(
        ontological_filepath="../../ontologies/pre/inputers/pima-diabetes.yaml"
    )
    diabetes = inputer.transform()
    leaner = Learner(
        ontological_filepath="../../ontologies/learners/RFC.pima.yaml"
    )
    leaner.train(
        diabetes, target=inputer.target, checkpoint="diabetesRandomForest1.ckp")
    learner.predict(diabetes, measure=True)
    learner.metrics

    assert leaner.target == inputer.target


# 9
