#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"
__coverage__ = 0.69

from pathlib import Path
import pytest
import numpy as np
from loguru import logger

# paso imports

from paso.base import Paso, Param, PasoError
from paso.pre.inputers import Inputers
from paso.learners.learners import Learners

fp = "../../parameters/base.yaml"
session = Paso(parameters_filepath=fp).startup()
# 0
def test_paso_param_file():
    assert session.parameters_filepath == fp


# 1
def test_paso_param_file_globa():
    assert session.parameters_filepath == Param.gfilepath


# 2
def test_sklearn_parm_global():
    assert Param.parameters_D["project"] == "HPKinetics/paso"


# 3
def test_learn_train_ontological_file_not_exist(flower):
    o = Learners(description_filepath="../../descriptions/learners/RandomForest.yaml")
    with pytest.raises(PasoError):
        o.train(flower) == o


# 4
def test_learn_train_bad_arg():
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/pima-diabetes.yaml"
    )
    dataset = inputer.transform()
    y = dataset[inputer.target].values
    X = dataset[dataset.columns.difference([inputer.target])]
    o = Learners(description_filepath="../../descriptions/learners/RFC.sm.yaml")
    with pytest.raises(TypeError):
        o.train(X, o) == o


#
def test_learn_train_no_y():
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/pima-diabetes.yaml"
    )
    dataset = inputer.transform()
    y = dataset[inputer.target].values
    X = dataset[dataset.columns.difference([inputer.target])]
    o = Learners(description_filepath="../../descriptions/learners/RFC.sm.yaml")
    with pytest.raises(IndexError):
        o.train(X) == o


# 6
def test_learn_train_kw_target_iris(flower):
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/pima-diabetes.yaml"
    )
    dataset = inputer.transform()
    y = dataset[inputer.target].values
    X = dataset[dataset.columns.difference([inputer.target])]
    o = Learners(description_filepath="../../descriptions/learners/RFC.sm.yaml")
    assert inputer.target == "Outcome"


# 7
def test_learn_train_kw_target_pima():
    inputer = Inputers(description_filepath="../../descriptions/pre/inputers/wine.yaml")
    dataset = inputer.transform()
    y = dataset[inputer.target].values
    X = dataset[dataset.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/RFC.yaml")
    learner.train(X, y, checkpoint="diabetesRandomForest1.ckp")
    assert learner.trained == True


# 8
def test_learn_model_name():
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/pima-diabetes.yaml"
    )
    dataset = inputer.transform()
    y = dataset[inputer.target].values
    X = dataset[dataset.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/RFC.yaml")
    learner.train(X, y, checkpoint="RandomForest.ckp")
    assert learner.model_name == "RandomForest"


# 9
def test_learn_model_type():
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/pima-diabetes.yaml"
    )
    dataset = inputer.transform()
    y = dataset[inputer.target].values
    X = dataset[dataset.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/RFC.yaml")
    learner.train(X, y, checkpoint="RandomForest.ckp")
    assert learner.model_type == "Classification"


# 10
def test_learn_train_model_nameXGBC():
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/pima-diabetes.yaml"
    )
    dataset = inputer.transform()
    y = dataset[inputer.target].values
    X = dataset[dataset.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/XGBC.yaml")
    learner.train(X, y, checkpoint="XGBClassifier.ckp")
    assert learner.model_name == "XGBClassifier"


# 11
def test_learn_train_model_name_LGBC():
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/pima-diabetes.yaml"
    )
    dataset = inputer.transform()
    y = dataset[inputer.target].values
    X = dataset[dataset.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/LGBC.yaml")
    learner.train(X, y, checkpoint="LGBMClassifier.ckp")
    assert learner.model_name == "LGBMClassifier"


# 12
def test_learn_train_predict_XGBC():
    inputer = Inputers(description_filepath="../../descriptions/pre/inputers/wine.yaml")
    dataset = inputer.transform()
    y = dataset[inputer.target].values
    X = dataset[dataset.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/XGBC.yaml")
    learner.train(X, y, checkpoint="LGBMClassifier.ckp")
    assert learner.predict(X).shape == (178,)


# 13
def test_learn_train_predict_prob_LGBC():
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/otto_group.yaml"
    )
    dataset = inputer.transform()
    y = dataset[inputer.target].values
    X = dataset[dataset.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/LGBC.yaml")
    learner.train(X, y, checkpoint="otto_group_LGBMC.ckp")
    assert learner.predict(X).shape == (61878,)


# 13
def test_learn_train_predict_Prob_XGBC():
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/pima-diabetes.yaml"
    )
    diabetes = inputer.transform()
    y = diabetes[inputer.target].values
    X = diabetes[diabetes.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/XGBC.yaml")
    learner.train(X, y, checkpoint="diabetesLGBMC.ckp")
    assert learner.predict_proba(X).shape == (768, 2)


# 14
def test_train_predict_prob_LGBC():
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/pima-diabetes.yaml"
    )
    diabetes = inputer.transform()
    y = diabetes[inputer.target].values
    X = diabetes[diabetes.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/LGBC.yaml")
    learner.train(X, y, checkpoint="diabetesLGBMC.ckp")
    assert learner.predict_proba(X).shape == (768, 2)


# 14b
def test_train_predict_prob_XGBC_creditdard():
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/creditcard.yaml"
    )
    creditdard = inputer.transform()
    y = creditdard[inputer.target].values
    X = creditdard[creditdard.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/XGBC.yaml")
    learner.train(X, y, checkpoint="creditdardLGBMC.ckp")
    assert learner.predict_proba(X).shape == (284807, 2)


# 14c
def test_train_predict_prob_XGBC_yeast3():
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/yeast3.yaml"
    )
    yeast3 = inputer.transform()
    y = yeast3[inputer.target].values
    X = yeast3[yeast3.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/XGBC.yaml")
    learner.train(X, y, checkpoint="yeast3LGBMC.ckp")
    assert learner.predict_proba(X).shape == (1484, 2)


# 14d
def test_train_predict_prob_LGBC_wine():
    inputer = Inputers(description_filepath="../../descriptions/pre/inputers/wine.yaml")
    wine = inputer.transform()
    y = wine[inputer.target].values
    X = wine[wine.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/LGBC.yaml")
    learner.train(X, y, target=inputer.target, checkpoint="wineLGBMC.ckp")
    assert learner.predict_proba(X).shape == (178, 3)


# 15
def test_predict_LGBC_no_fit():
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/pima-diabetes.yaml"
    )
    diabetes = inputer.transform()
    learner = Learners(description_filepath="../../descriptions/learners/LGBC.yaml")
    # learner.train(
    #     diabetes, target=inputer.target, checkpoint="diabetesLGBMC.ckp"
    # )
    X = diabetes
    X_train = X[X.columns.difference([inputer.target])]
    with pytest.raises(PasoError):
        assert learner.predict(X_train).shape == (768, 2)


# 15
def test_predict_Prob_error():
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/pima-diabetes.yaml"
    )
    diabetes = inputer.transform()
    learner = Learners(description_filepath="../../descriptions/learners/LGBC.yaml")
    # learner.train(
    #     diabetes, target=inputer.target, checkpoint="diabetesLGBMC.ckp"
    # )
    X = diabetes
    X_train = X[X.columns.difference([inputer.target])]
    with pytest.raises(PasoError):
        assert learner.predict_proba(X_train).shape == (768, 2)


# 16
def test_evaluate_pima():
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/pima-diabetes.yaml"
    )
    dataset = inputer.transform()
    y = dataset[inputer.target].values
    X = dataset[dataset.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/LGBC.yaml")
    learner.train(X, y, checkpoint="pima_LGBMC.ckp")

    assert len(learner.evaluate(X, y).keys()) == 7


# 17
def test_evaluate_otto_group():
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/otto_group.yaml"
    )
    dataset = inputer.transform()
    y = dataset[inputer.target].values
    X = dataset[dataset.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/XGBC.yaml")
    learner.train(X, y, checkpoint="pima_LGBMC.ckp")

    assert len(learner.evaluate(X, y).keys()) == 7


# 18
def test_learner_cross_validate_LGBC():
    inputer = Inputers(
        description_filepath="../../descriptions/pre/inputers/pima-diabetes.yaml"
    )
    dataset = inputer.transform()
    y = dataset[inputer.target].values
    X = dataset[dataset.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/LGBC.yaml")
    learner.train(X, y, checkpoint="pima_LGBMC.ckp")
    learner.cross_validate(
        X,
        y,
        description_filepath="../../descriptions/learners/Cross_validation_classification.yaml",
    )

    assert len(learner.evaluate(X, y).keys()) == 8

# 19
def test_learner_cross_validate_RFC_iris_milticlaas():
    inputer = Inputers(description_filepath="../../descriptions/pre/inputers/iris.yaml")
    dataset = inputer.transform()
    y = dataset[inputer.target].values
    X = dataset[dataset.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/RFC.yaml")
    learner.train(X, y, checkpoint="pima_LGBMC.ckp")
    learner.cross_validate(
        X,
        y,
        description_filepath="../../descriptions/learners/Cross_validation_classification.yaml",
    )

    assert learner.cv == 5


# 20
def test_learner_cross_validate_RFC_iris_milticlass_evaluate_AO():
    inputer = Inputers(description_filepath="../../descriptions/pre/inputers/iris.yaml")
    dataset = inputer.transform()
    y = dataset[inputer.target].values
    X = dataset[dataset.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/RFC.yaml")
    learner.train(X, y, checkpoint="iris)RC.ckp")
    learner.cross_validate(
        X,
        y,
        description_filepath="../../descriptions/learners/Cross_validation_classification.yaml",
    )

    assert learner.evaluate(X, y)["AOC"] == "must be binary class"


# 21
def test_learner_cross_validate_RFC_iris_milticlass_evaluate_test_accuracy():
    inputer = Inputers(description_filepath="../../descriptions/pre/inputers/iris.yaml")
    dataset = inputer.transform()
    y = dataset[inputer.target].values
    X = dataset[dataset.columns.difference([inputer.target])]
    learner = Learners(description_filepath="../../descriptions/learners/RFC.yaml")
    learner.train(X, y, checkpoint="pima_LGBMC.ckp")
    score = learner.cross_validate(
        X,
        y,
        description_filepath="../../descriptions/learners/Cross_validation_classification.yaml",
    )

    assert score["mean"]["test_accuracy"] >= 0.95
